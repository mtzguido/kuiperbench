import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
from _kuiper_dwconv2d import kuiper_dwconv2d
import torch.nn as nn

class ModelNew(nn.Module):
    """KB L1 #85: depthwise 2D conv, asymmetric input, asymmetric (kh, kw) kernel."""
    def __init__(self, in_channels, out_channels, kernel_size_h, kernel_size_w,
                 stride_h=1, stride_w=1, padding_h=0, padding_w=0,
                 dilation_h=1, dilation_w=1, groups=1, bias=False):
        super().__init__()
        # KB harness sets groups=in_channels in get_init_inputs.
        assert groups == in_channels, \
            "Kuiper depthwise: only groups == in_channels supported"
        assert out_channels == in_channels, \
            "Kuiper depthwise: only out_channels == in_channels supported"
        self.conv2d = nn.Conv2d(in_channels, in_channels,
                                (kernel_size_h, kernel_size_w),
                                stride=(stride_h, stride_w),
                                padding=(padding_h, padding_w),
                                dilation=(dilation_h, dilation_w),
                                groups=in_channels, bias=bias)
        self._stride = (stride_h, stride_w)
        self._padding = (padding_h, padding_w)
        self._dilation = (dilation_h, dilation_w)
        self._groups = in_channels

    def forward(self, x):
        w = self.conv2d.weight.contiguous().to(x.device).to(torch.float32)
        b = None
        if self.conv2d.bias is not None:
            b = self.conv2d.bias.contiguous().to(x.device).to(torch.float32)
        return kuiper_dwconv2d(x, w, b,
                               stride=self._stride, padding=self._padding,
                               dilation=self._dilation, groups=self._groups)
