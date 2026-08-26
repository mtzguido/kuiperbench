import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
from _kuiper_conv2d_general import kuiper_conv2d_general
import torch.nn as nn

class ModelNew(nn.Module):
    """KB L1 #55: asymmetric input, square kernel."""
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=False):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels,
                                (kernel_size, kernel_size),
                                stride=stride, padding=padding,
                                dilation=dilation, groups=groups, bias=bias)
        self._stride = stride
        self._padding = padding
        self._dilation = dilation
        self._groups = groups

    def forward(self, x):
        w = self.conv2d.weight.contiguous().to(x.device).to(torch.float32)
        b = None
        if self.conv2d.bias is not None:
            b = self.conv2d.bias.contiguous().to(x.device).to(torch.float32)
        return kuiper_conv2d_general(x, w, b,
                                     stride=self._stride, padding=self._padding,
                                     dilation=self._dilation, groups=self._groups)
