import torch
import torch.nn as nn
import os
import shutil
from torch.utils.cpp_extension import load

_dir = os.path.dirname(os.path.abspath(__file__))
_kuiper_root = os.path.abspath(os.path.join(_dir, '..', '..', '..', '..'))
_obj_dir = os.path.join(_kuiper_root, 'obj')
_kuiper_home = os.environ.get('KUIPER_HOME', os.path.join(_kuiper_root, '.kuiper'))
_include_dir = os.path.join(_kuiper_home, 'include')
_kbench_include_dir = os.path.join(_kuiper_root, 'include')
_bridge = os.path.join(_dir, 'kuiper_argmaxreducedim_bridge.cu')

# Mirror the build-dir layout used by #49/#53 so torch's cpp_extension
# picks up the extracted .cu by include path.  No regex rewrite needed
# here: neg_inf is inlined as INFINITY by the kernel (see Fmax.fsti's
# inline_for_extraction noextract pos_inf/neg_inf).
_build_dir = os.path.join(_dir, 'build')
os.makedirs(_build_dir, exist_ok=True)
_src_cu = os.path.join(_obj_dir, 'Kuiper_KB_ArgmaxReduceDim.cu')
_dst_cu = os.path.join(_build_dir, 'Kuiper_KB_ArgmaxReduceDim.cu')
shutil.copyfile(_src_cu, _dst_cu)

kuiper_argmax = load(
    name="kuiper_argmaxreducedim_51",
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
        if self.dim != 1:
            raise ValueError("Kuiper argmax-reduction supports dim=1")
        return kuiper_argmax.kuiper_argmaxreduce_dim1(x)
