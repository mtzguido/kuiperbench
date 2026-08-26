import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
from _kuiper_convt2d_general import kuiper_convt2d_general
import torch.nn as nn

class ModelNew(nn.Module):
    """KB L1 #81: ConvTranspose2D dilated/padded/strided (Kuiper-verified)."""
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation, bias=bias)

    def forward(self, x):
        ct = self.conv_transpose2d
        w = ct.weight.contiguous().to(x.device).to(torch.float32)
        b = None
        if ct.bias is not None:
            b = ct.bias.contiguous().to(x.device).to(torch.float32)
        return kuiper_convt2d_general(
            x, w, b,
            stride=ct.stride, padding=ct.padding,
            output_padding=ct.output_padding,
            dilation=ct.dilation, groups=ct.groups)
