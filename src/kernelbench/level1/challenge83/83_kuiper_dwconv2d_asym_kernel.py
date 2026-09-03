import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
from _kuiper_dwconv2d import kuiper_dwconv2d
import torch.nn as nn

class ModelNew(nn.Module):
    """KB L1 #83: depthwise 2D conv, square input, asymmetric (k, 1) kernel."""
    def __init__(self, in_channels, kernel_size, stride=1, padding=0,
                 dilation=1, bias=False):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels, in_channels,
                                kernel_size=(kernel_size, 1),
                                stride=stride, padding=padding,
                                dilation=dilation, groups=in_channels,
                                bias=bias)
        self._stride = stride
        self._padding = padding
        self._dilation = dilation
        self._groups = in_channels

    def forward(self, x):
        return kuiper_dwconv2d(x, self.conv2d.weight, self.conv2d.bias,
                               stride=self._stride, padding=self._padding,
                               dilation=self._dilation, groups=self._groups)
