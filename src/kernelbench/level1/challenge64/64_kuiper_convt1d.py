import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
from _kuiper_convt1d_general import kuiper_convt1d_general
import torch.nn as nn

class ModelNew(nn.Module):
    """KB L1 #64: ConvTranspose1D (Kuiper-verified, general parameters).

    Defaults match the KernelBench Model: stride=1, padding=0,
    output_padding=0, dilation=1, groups=1, bias=False.  Implemented by
    reusing the verified ConvT2D kernel via a 1-axis embedding (H=Kh=1).
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, output_padding=0,
                 dilation=1, groups=1, bias=False):
        super().__init__()
        self.conv1d_transpose = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding,
            dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        ct = self.conv1d_transpose
        w = ct.weight.contiguous().to(x.device).to(torch.float32)
        b = None
        if ct.bias is not None:
            b = ct.bias.contiguous().to(x.device).to(torch.float32)
        return kuiper_convt1d_general(
            x, w, b,
            stride=ct.stride[0], padding=ct.padding[0],
            output_padding=ct.output_padding[0],
            dilation=ct.dilation[0], groups=ct.groups)
