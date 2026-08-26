import torch
import torch.nn as nn
import os
import re
from torch.utils.cpp_extension import load

_dir = os.path.dirname(os.path.abspath(__file__))
_kuiper_root = os.path.abspath(os.path.join(_dir, '..', '..', '..', '..'))
_obj_dir = os.path.join(_kuiper_root, 'obj')
_kuiper_home = os.environ.get('KUIPER_HOME', os.path.join(_kuiper_root, '.kuiper'))
_include_dir = os.path.join(_kuiper_home, 'include')
_kbench_include_dir = os.path.join(_kuiper_root, 'include')
_bridge = os.path.join(_dir, 'kuiper_minreducedim_bridge.cu')

# The extracted .cu declares [extern float Kuiper_Math_Fmin_pos_inf;] at
# file scope and reads it inside a __global__ kernel.  Strip the host
# extern so the bridge can supply a __device__ definition of the same
# symbol.  Cache the rewritten file under build/ so torch's cpp_extension
# can pick it up like an ordinary source.
_build_dir = os.path.join(_dir, 'build')
os.makedirs(_build_dir, exist_ok=True)
_src_cu = os.path.join(_obj_dir, 'Kuiper_KB_MinReduceDim.cu')
_fixed_cu = os.path.join(_build_dir, 'Kuiper_KB_MinReduceDim.cu')
with open(_src_cu) as f:
    _content = f.read()
_content = re.sub(r'^\s*extern\s+float\s+Kuiper_Math_Fmin_pos_inf\s*;\s*$',
                  '/* host extern of pos_inf stripped; bridge defines a __device__ symbol */',
                  _content, flags=re.MULTILINE)
with open(_fixed_cu, 'w') as f:
    f.write(_content)

kuiper_minreduce = load(
    name="kuiper_minreducedim_53",
    sources=[_bridge],
    extra_include_paths=[_build_dir, _include_dir, _kbench_include_dir, _obj_dir],
    extra_cuda_cflags=['-DKUIPER_CFG_TENSORCORES=0'],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim == 1 and x.dim() == 3 and x.dtype == torch.float32 and x.is_cuda:
            return kuiper_minreduce.kuiper_minreduce_dim1(x)
        return torch.min(x, dim=self.dim)[0]
