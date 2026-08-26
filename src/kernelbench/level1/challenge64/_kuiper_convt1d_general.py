import os
import torch
from torch.utils.cpp_extension import load

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_KUIPER_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..', '..', '..'))
_OBJ_DIR = os.path.join(_KUIPER_ROOT, 'obj')
_KUIPER_HOME = os.environ.get('KUIPER_HOME', os.path.join(_KUIPER_ROOT, '.kuiper'))
_INCLUDE_DIR = os.path.join(_KUIPER_HOME, 'include')
_KBENCH_INCLUDE_DIR = os.path.join(_KUIPER_ROOT, 'include')

_module = load(
    name="kuiper_convt1d_general_64",
    sources=[os.path.join(_THIS_DIR, 'kuiper_convt1d_general_bridge.cu')],
    extra_include_paths=[_INCLUDE_DIR, _KBENCH_INCLUDE_DIR, _OBJ_DIR, _THIS_DIR],
    extra_cuda_cflags=['-DKUIPER_CFG_TENSORCORES=0'],
    verbose=True,
)

def kuiper_convt1d_general(x, w, bias=None, stride=1, padding=0,
                           output_padding=0, dilation=1, groups=1):
    return _module.kuiper_convt1d_general(
        x, w, bias, int(stride), int(padding), int(output_padding),
        int(dilation), int(groups))
