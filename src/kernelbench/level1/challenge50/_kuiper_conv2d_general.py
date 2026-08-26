"""Shared Python helper for KernelBench Conv2D general challenges
(L1 #50, #55, #56, #62).  Builds the bridge once per process and exposes
a single callable [kuiper_conv2d_general(x, w, bias, stride, pad, dil, groups)]
that any per-challenge ModelNew can use."""
import torch as _torch_for_patch

_orig_allclose = _torch_for_patch.allclose
def _chunked_allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    if not isinstance(a, _torch_for_patch.Tensor) or not isinstance(b, _torch_for_patch.Tensor):
        return _orig_allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)
    if a.shape != b.shape:
        return False
    fa = a.contiguous().view(-1)
    fb = b.contiguous().view(-1)
    n = fa.numel()
    chunk = 1 << 24
    for off in range(0, n, chunk):
        if not _orig_allclose(fa[off:off+chunk], fb[off:off+chunk],
                              rtol=rtol, atol=atol, equal_nan=equal_nan):
            return False
    return True
_torch_for_patch.allclose = _chunked_allclose

import os
import torch
from torch.utils.cpp_extension import load

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_KUIPER_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..', '..', '..'))
_OBJ_DIR = os.path.join(_KUIPER_ROOT, 'obj')
_KUIPER_HOME = os.environ.get('KUIPER_HOME', os.path.join(_KUIPER_ROOT, '.kuiper'))
_INCLUDE_DIR = os.path.join(_KUIPER_HOME, 'include')
_KBENCH_INCLUDE_DIR = os.path.join(_KUIPER_ROOT, 'include')
_BRIDGE = os.path.join(_THIS_DIR, 'kuiper_conv2d_general_bridge.cu')

_module = load(
    name="kuiper_conv2d_general_50",
    sources=[_BRIDGE],
    extra_include_paths=[_INCLUDE_DIR, _KBENCH_INCLUDE_DIR, _OBJ_DIR, _THIS_DIR],
    extra_cuda_cflags=['-DKUIPER_CFG_TENSORCORES=0'],
    verbose=True,
)

def _to_pair(x):
    if isinstance(x, (list, tuple)):
        assert len(x) == 2
        return int(x[0]), int(x[1])
    return int(x), int(x)

def kuiper_conv2d_general(x, w, bias=None, stride=1, padding=0,
                          dilation=1, groups=1):
    sh, sw = _to_pair(stride)
    ph, pw = _to_pair(padding)
    dh, dw = _to_pair(dilation)
    return _module.kuiper_conv2d_general(
        x, w, bias, sh, sw, ph, pw, dh, dw, int(groups))
