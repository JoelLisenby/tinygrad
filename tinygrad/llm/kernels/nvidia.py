from __future__ import annotations
import functools, math
from typing import Callable
from tinygrad import Tensor, UOp
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.helpers import prod
from tinygrad.uop.ops import AxisType, KernelInfo, Ops
from tinygrad.llm.kernels.amd import Linear as AMDLinear, Q4_K, Q5_K, Q6_K, IQ4_XS, GGML_BLOCK_SIZE, Q8_GROUP_SIZE, \
  Q4_WORDS, Q5_WORDS, Q6_WORDS, IQ4_WORDS, WARP_SIZE, _half, _iq4_scales

def cuda_custom_kernels_supported(device:str|tuple[str, ...]|None) -> bool:
  if isinstance(device, tuple): device = device[0]
  return device is not None and device.split(":")[0] in ("CUDA", "NV")

def warp_reduce(val:UOp, maximum:bool=False, full_wave:bool=False) -> UOp:
  # CUDA warp is 32 threads; shfl_xor covers the same offsets as the AMD ds_swizzle tree
  for offset in ((16, 8, 4, 2, 1) if full_wave else (8, 4, 2, 1)):
    if val.op is Ops.INDEX and val.addrspace == AddrSpace.REG: val = val.load()
    other = UOp(Ops.CUSTOM, src=(val,), arg=(f"__shfl_xor_sync(0xffffffff, {{0}}, {offset})", val.dtype))
    val = val.maximum(other) if maximum else val + other
  return val

def _load(ptr:UOp, lanes:int|None=None) -> UOp:
  assert ptr.op is Ops.INDEX
  if lanes is None: return ptr.load()
  buf, coords = ptr.src[0], ptr.src[1:]
  idx = sum((coord*math.prod(buf.shape[i+1:]) for i,coord in enumerate(coords)), UOp.const(0))
  return UOp(Ops.SHRINK, src=(buf.flatten(), idx, UOp.const(lanes))).load(dtype=ptr.dtype)

def _dp4a(a:UOp, b:UOp, c:UOp) -> UOp:
  return UOp(Ops.CUSTOM, src=(a.cast(dtypes.int32), b.cast(dtypes.int32), c), arg=("__dp4a({0}, {1}, {2})", dtypes.int32))

def _perm_index(sel:UOp) -> UOp:
  # AMD perm selector is one nibble per byte; CUDA __byte_perm packs those nibbles into one word
  sel = sel.cast(dtypes.uint32)
  return (sel & 15) | (((sel >> 8) & 15) << 4) | (((sel >> 16) & 15) << 8) | (((sel >> 24) & 15) << 12)

def _byte_perm(a:UOp, b:UOp, selectors:UOp) -> UOp:
  # CUDA __byte_perm(a,b): a=bytes 0-3, b=bytes 4-7. AMD perm(src0,src1) is the other way on some docs;
  # IQ4_XS only matches with (b, a).
  src = tuple(x.cast(dtypes.uint32) for x in (b, a, _perm_index(selectors)))
  return UOp(Ops.CUSTOMI, src=src, arg=("__byte_perm({}, {}, {})", dtypes.uint32))

def _iq4_bytes(packed:UOp, shift:int) -> UOp:
  selectors = (packed >> shift) & 0x0f0f0f0f
  low = _byte_perm(UOp.const(0xf6eaddcf, dtypes.uint32), UOp.const(0xbfad9881, dtypes.uint32), selectors)
  high = _byte_perm(UOp.const(0x71594535, dtypes.uint32), UOp.const(0x26190d01, dtypes.uint32), selectors & 0x07070707)
  return _byte_perm(high, low, 0x03020100 | ((selectors & 0x08080808) >> 1))

def _q5_scales(raw:UOp, base:UOp, subgroup:UOp) -> tuple[UOp, UOp, UOp, UOp]:
  w1, w2, w3 = _load(raw[base+1]), _load(raw[base+2]), _load(raw[base+3])
  sb = (subgroup & 3) * 8
  byte1, byte2, byte3 = (w1 >> sb) & 255, (w2 >> sb) & 255, (w3 >> sb) & 255
  scale = (subgroup < 4).where(byte1 & 63, (byte3 & 15) | ((byte1 >> 6) << 4))
  minimum = (subgroup < 4).where(byte2 & 63, (byte3 >> 4) | ((byte2 >> 6) << 4))
  d, dmin = (raw[base] & 0xffff).cast(dtypes.uint16), (raw[base] >> 16).cast(dtypes.uint16)
  return _half(d), _half(dmin), scale.float(), minimum.float()

class Linear(AMDLinear):
  def __call__(self, x:Tensor) -> Tensor:
    supported = self.use_custom_quant and cuda_custom_kernels_supported(self.weight.device)
    if self.ggml_type is None and supported:
      self.set_quantized(self.weight)
      if self.ggml_type is None:
        if self.weight.dtype in (dtypes.half, dtypes.float, dtypes.bfloat16) and self.out_features <= 2048 \
          and self.in_features % (WARP_SIZE*4) == 0:
          numel, max_shape = x.numel(), x.max_shape
          if isinstance(numel, int) or prod(max_shape) // self.in_features <= 32:
            out = f16_gemv(self, x if isinstance(numel, int) else x.pad_to(max_shape))
            return out if isinstance(numel, int) else out.shrink(tuple((0, s) for s in (*x.shape[:-1], self.out_features)))
        self.use_custom_quant = supported = False
    if self.ggml_type in (Q4_K, Q5_K, Q6_K, IQ4_XS) and supported:
      if isinstance(x.numel(), int): return q8_linear(self, x)
      out = q8_linear(self, x.pad_to(x.max_shape))
      return out.shrink(tuple((0, s) for s in (*x.shape[:-1], self.out_features)))
    return super().__call__(x)

@functools.cache
def _q8_quantize_kernel(q:UOp, scale:UOp, xsum:UOp, x:UOp, tokens:int, in_features:int) -> UOp:
  groups = in_features//Q8_GROUP_SIZE
  token_group, lane = UOp.range(tokens*groups, 0, axis_type=AxisType.GLOBAL), UOp.range(32, 1, axis_type=AxisType.LOCAL)
  token, group = token_group//groups, token_group%groups
  x = x.reshape(tokens, groups, 32)
  group_scale = (warp_reduce(x[token, group, lane].float().abs(), maximum=True, full_wave=True) / 127).maximum(1e-8)
  word_lane = lane.minimum(7)
  xs = tuple(x[token, group, word_lane*4+i].float() for i in range(4))
  qs = tuple((v/group_scale).round().clip(-127, 127).cast(dtypes.int8) for v in xs)
  word = sum((v.cast(dtypes.uint8).cast(dtypes.uint32) << (i*8) for i, v in enumerate(qs)), UOp.const(0, dtypes.uint32))
  part = (lane < 8).where(sum((v.cast(dtypes.int32) for v in qs), UOp.const(0, dtypes.int32)), UOp.const(0, dtypes.int32))
  gsum = [warp_reduce(((lane & 4).eq(h*4)).where(part, UOp.const(0, dtypes.int32)), full_wave=True) for h in range(2)]
  store_half = (lane & 4) >> 2
  stores = (q[token, group, lane.valid(lane < 8)].store(word),
            UOp.group(scale[token, group.valid(lane.eq(0))].store(group_scale),
                      xsum[token, group, store_half.valid(lane.eq(0) | lane.eq(4))].store(
                        store_half.eq(0).where(gsum[0].float(), gsum[1].float()))))
  return UOp.group(*stores).end(token_group, lane).sink(arg=KernelInfo(name="q8_quantize", opts_to_apply=()))

def q8_quantize(x:Tensor, tokens:int, in_features:int) -> tuple[Tensor, Tensor, Tensor]:
  groups = in_features//Q8_GROUP_SIZE
  q = Tensor.empty(tokens, groups, 8, dtype=dtypes.uint32, device=x.device)
  scale = Tensor.empty(tokens, groups, dtype=dtypes.float32, device=x.device)
  xsum = Tensor.empty(tokens, groups, 2, dtype=dtypes.float32, device=x.device)
  q, scale, xsum = Tensor.custom_kernel(q, scale, xsum, x, fxn=functools.partial(_q8_quantize_kernel, tokens=tokens, in_features=in_features))[:3]
  return q, scale, xsum

def _decode_linear(out:UOp, out_features:int, group_count:int, group_dot, name:str) -> UOp:
  chunks = out.shape[2]
  token_output = UOp.range(out.shape[0]*out_features, 0, axis_type=AxisType.GLOBAL)
  chunk, lane = UOp.range(chunks, 1, axis_type=AxisType.GLOBAL), UOp.range(32, 2, axis_type=AxisType.LOCAL)
  token, output = token_output // out_features, token_output % out_features
  group = (lane+chunk*32).minimum(group_count-1)
  value = group_dot(token, output, group) if chunks*32 == group_count else \
    (lane+chunk*32 < group_count).where(group_dot(token, output, group), UOp.const(0, dtypes.float32))
  total = warp_reduce(value, full_wave=True)
  return out[token, output, chunk.valid(lane.eq(0))].store(total.cast(out.dtype)).end(token_output, chunk, lane).sink(
    arg=KernelInfo(name=name, opts_to_apply=()))

@functools.cache
def _quant_decode_kernel(out:UOp, raw:UOp, xq:UOp, xd:UOp, xs:UOp, out_features:int, in_features:int, ggml_type:int) -> UOp:
  group_count = in_features // Q8_GROUP_SIZE
  def group_dot(token:UOp, output:UOp, group:UOp) -> UOp:
    block, subgroup = group // 8, group % 8
    xwords = _load(xq[token, group, 0], 8)
    if ggml_type in (Q4_K, Q5_K):
      base = (output * in_features//GGML_BLOCK_SIZE + block) * (Q4_WORDS if ggml_type == Q4_K else Q5_WORDS)
      qs_base, dot = base + (4 if ggml_type == Q4_K else 12) + (subgroup//2)*8, UOp.const(0, dtypes.int32)
      qs_pair = (_load(raw[qs_base], 4), _load(raw[qs_base+4], 4))
      if ggml_type == Q5_K: qh_pair = (_load(raw[base+4], 4), _load(raw[base+8], 4))
      for word_idx in range(8):
        word = (qs_pair[word_idx//4][word_idx%4] >> ((subgroup&1)*4).cast(dtypes.uint32)) & 0x0f0f0f0f
        if ggml_type == Q5_K: word |= ((qh_pair[word_idx//4][word_idx%4] >> subgroup.cast(dtypes.uint32)) & 0x01010101) << 4
        dot = _dp4a(word, xwords[word_idx], dot)
      d, dmin, scale, minimum = _q5_scales(raw, base, subgroup)
      gsum = xs[token, group, 0].load() + xs[token, group, 1].load()
      return (dot.float()*d*scale - gsum*dmin*minimum) * xd[token, group]
    if ggml_type == IQ4_XS:
      base = (output * in_features//GGML_BLOCK_SIZE + block) * IQ4_WORDS
      dot = UOp.const(0, dtypes.int32)
      for word_idx in range(8):
        packed = _load(raw[base + 2 + subgroup*4 + word_idx%4])
        dot = _dp4a(_iq4_bytes(packed, 4*(word_idx//4)), xwords[word_idx], dot)
      d, scale = _iq4_scales(raw, base, subgroup)
      return dot.float() * xd[token, group] * d * scale
    base = (output*in_features//GGML_BLOCK_SIZE+block)*Q6_WORDS
    lows = tuple(_load(raw[base + (subgroup//4)*16 + (subgroup%2)*8 + half*4], 4) for half in range(2))
    highs = tuple(_load(raw[base + 32 + (subgroup//4)*8 + half*4], 4) for half in range(2))
    dots = [UOp.const(0, dtypes.int32)] * 2
    for word_idx in range(8):
      within = (subgroup*32 + word_idx*4)%128
      low = lows[word_idx//4][word_idx%4] >> ((within//64)*4).cast(dtypes.uint32)
      high = highs[word_idx//4][word_idx%4] >> ((within//32)*2).cast(dtypes.uint32)
      word = (low & 0x0f0f0f0f) | ((high & 0x03030303) << 4)
      dots[word_idx//4] = _dp4a(word, xwords[word_idx], dots[word_idx//4])
    scales = [((raw[base + 48 + (subgroup*2+i)//4] >> (((subgroup*2+i)%4)*8).cast(dtypes.uint32)) & 255)
              .cast(dtypes.uint8).bitcast(dtypes.int8).float() for i in range(2)]
    gsum = [xs[token, group, i].load() * 32 for i in range(2)]
    return ((dots[0].float() - gsum[0])*scales[0] + (dots[1].float() - gsum[1])*scales[1]) * xd[token, group] * _half(raw[base+52] & 0xffff)
  names = {Q4_K: "linear_q4_k", Q5_K: "linear_q5_k", IQ4_XS: "linear_iq4_xs", Q6_K: "linear_q6"}
  return _decode_linear(out, out_features, group_count, group_dot, names[ggml_type])

def q8_linear(layer:Linear, x:Tensor) -> Tensor:
  assert layer.ggml_type in (Q4_K, Q5_K, Q6_K, IQ4_XS)
  tokens = int(x.numel()) // layer.in_features
  raw, out_features, in_features = layer.weight.uop.buf_uop, layer.out_features, layer.in_features
  def run(fxn:Callable[..., UOp], out:UOp, *srcs:UOp) -> Tensor:
    all_srcs = (out,)+srcs
    params = tuple(UOp.placeholder_like(src, slot=i) for i,src in enumerate(all_srcs))
    kernel = fxn(*params, out_features=out_features, in_features=in_features).call(*all_srcs)
    result = Tensor(out.after(kernel))
    if len(result.shape) == 3: result = result.sum(-1)
    result = result.reshape(*x.shape[:-1], out_features)
    return result if layer.bias is None else result + layer.bias
  xq_, xd, xs = q8_quantize(x, tokens, in_features)
  decode = functools.partial(_quant_decode_kernel, ggml_type=layer.ggml_type)
  out = Tensor.empty(tokens, out_features, (in_features+1023)//1024, dtype=dtypes.float32, device=x.device).uop
  return run(decode, out, raw, xq_.uop, xd.uop, xs.uop)

def _view_back(t:Tensor) -> Tensor:
  uop = t.uop
  while uop.op is Ops.CAST: uop = uop.src[0]
  return Tensor(uop).reshape(t.shape)

@functools.cache
def _f16_gemv_kernel(out:UOp, w:UOp, x:UOp, *rest:UOp, in_features:int, out_features:int, tokens:int) -> UOp:
  bias: UOp|None = rest[0] if rest else None
  token, out_row = UOp.range(tokens, 0, AxisType.GLOBAL), UOp.range(out_features, 1, AxisType.GLOBAL)
  lane = UOp.range(WARP_SIZE, 2, axis_type=AxisType.LOCAL)
  per, val_chunk = in_features // (WARP_SIZE * 4), 4
  assert per * WARP_SIZE * val_chunk == in_features
  w, x = w.reshape((out_features, per, WARP_SIZE*val_chunk)), x.reshape((tokens, per, WARP_SIZE*val_chunk))
  acc = UOp.const(0, dtypes.float32)
  for i in range(per):
    for j in range(val_chunk):
      acc = acc + w[out_row, i, lane*val_chunk + j].load().float() * x[token, i, lane*val_chunk + j].load().float()
  total = warp_reduce(acc, full_wave=True)
  if bias is not None: total = total + bias[token, out_row].load().float()
  return out[token, out_row.valid(lane.eq(0))].store(total).end(token, out_row, lane).sink(arg=KernelInfo(name="linear_f16_gemv", opts_to_apply=()))

def f16_gemv(layer:Linear, x:Tensor) -> Tensor:
  tokens = prod(x.shape[:-1])
  assert isinstance(tokens, int)
  weight, x = _view_back(layer.weight), x.contiguous() if x.dtype == dtypes.half else x.cast(dtypes.half).contiguous()
  out = Tensor.empty(tokens, layer.out_features, dtype=dtypes.float32, device=x.device)
  fxn = functools.partial(_f16_gemv_kernel, in_features=layer.in_features, out_features=layer.out_features, tokens=tokens)
  srcs = (out, weight.reshape(-1), x.reshape(tokens, layer.in_features)) + (() if layer.bias is None else (_view_back(layer.bias),))
  return Tensor.custom_kernel(*srcs, fxn=fxn)[0].reshape(*x.shape[:-1], layer.out_features)
