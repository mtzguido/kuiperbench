import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
from _kuiper_conv3d_general import kuiper_conv3d_general
import torch.nn as nn

class ModelNew(nn.Module):
    """KB L1 #59: 3D conv, asymmetric input, square (kh,kh,1) kernel.

    Matches the upstream Model's Conv3d construction:
        nn.Conv3d(in_channels, out_channels, (kernel_size, kernel_size, 1), ...)
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=False):
        super().__init__()
        k = (kernel_size, kernel_size, 1)
        self.conv3d = nn.Conv3d(in_channels, out_channels, k,
                                stride=stride, padding=padding,
                                dilation=dilation, groups=groups, bias=bias)
        self._stride = stride
        self._padding = padding
        self._dilation = dilation
        self._groups = groups

    def forward(self, x):
        return kuiper_conv3d_general(x, self.conv3d.weight, self.conv3d.bias,
                                     stride=self._stride, padding=self._padding,
                                     dilation=self._dilation, groups=self._groups)
