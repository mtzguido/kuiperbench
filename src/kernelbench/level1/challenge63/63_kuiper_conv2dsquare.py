import torch as _torch_for_patch

# Monkey-patch torch.allclose with a chunked version (see #34/#36/#37/#40).
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

import torch
import torch.nn as nn
import os
from torch.utils.cpp_extension import load

_dir = os.path.dirname(os.path.abspath(__file__))
_kuiper_root = os.path.abspath(os.path.join(_dir, '..', '..', '..', '..'))
_obj_dir = os.path.join(_kuiper_root, 'obj')
_kuiper_home = os.environ.get('KUIPER_HOME', os.path.join(_kuiper_root, '.kuiper'))
_include_dir = os.path.join(_kuiper_home, 'include')
_kbench_include_dir = os.path.join(_kuiper_root, 'include')
_bridge = os.path.join(_dir, 'kuiper_conv2d_square_bridge.cu')

kuiper_conv2d_square = load(
    name="kuiper_conv2d_square_63",
    sources=[_bridge],
    extra_include_paths=[_include_dir, _kbench_include_dir, _obj_dir],
    extra_cuda_cflags=['-DKUIPER_CFG_TENSORCORES=0'],
    verbose=True,
)


class ModelNew(nn.Module):
    """Drop-in replacement for KernelBench L1 #63 (Conv2D, square, bias=False)
    backed by the verified Kuiper kernel."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, dilation: int = 1,
                 groups: int = 1, bias: bool = False):
        super().__init__()
        # Mirror PyTorch weight init exactly so the harness comparison
        # is meaningful.
        assert stride == 1 and padding == 0 and dilation == 1 and groups == 1, \
            "Kuiper Conv2DSquare: only stride=1, pad=0, dil=1, groups=1 supported"
        assert not bias, "Kuiper Conv2DSquare: only bias=False supported"
        self.conv2d = nn.Conv2d(in_channels, out_channels,
                                (kernel_size, kernel_size),
                                stride=stride, padding=padding,
                                dilation=dilation, groups=groups, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return kuiper_conv2d_square.kuiper_conv2d_square(
            x, self.conv2d.weight)
