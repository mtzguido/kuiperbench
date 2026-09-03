import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
import torch.nn as nn
from torch.utils.cpp_extension import load

_orig_allclose = torch.allclose
def _chunked_allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
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
torch.allclose = _chunked_allclose

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_KUIPER_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..', '..', '..'))
_OBJ_DIR = os.path.join(_KUIPER_ROOT, 'obj')
_KUIPER_HOME = os.environ.get('KUIPER_HOME', os.path.join(_KUIPER_ROOT, '.kuiper'))
_INCLUDE_DIR = os.path.join(_KUIPER_HOME, 'include')
_KBENCH_INCLUDE_DIR = os.path.join(_KUIPER_ROOT, 'include')

_module = load(
    name="kuiper_dwsep_86",
    sources=[os.path.join(_THIS_DIR, 'kuiper_dwsep_bridge.cu')],
    extra_include_paths=[_INCLUDE_DIR, _KBENCH_INCLUDE_DIR, _OBJ_DIR, _THIS_DIR],
    extra_cuda_cflags=['-DKUIPER_CFG_TENSORCORES=0'],
    verbose=True,
)


class ModelNew(nn.Module):
    """KB L1 #86: depthwise-separable 2D conv = depthwise(C,C,K,...) then pointwise(C,Cout,1)."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, bias=False):
        super().__init__()
        assert dilation == 1, "Kuiper dwsep: only dilation=1 supported"
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   stride=stride, padding=padding,
                                   dilation=dilation, groups=in_channels,
                                   bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                   bias=bias)
        self._stride = stride
        self._padding = padding
        self._dilation = dilation

    def forward(self, x):
        return _module.kuiper_dwsep(
                                    x, self.depthwise.weight,
                                    self.pointwise.weight,
                                    self.depthwise.bias, self.pointwise.bias,
                                    int(self._stride), int(self._padding),
                                    int(self._dilation))
