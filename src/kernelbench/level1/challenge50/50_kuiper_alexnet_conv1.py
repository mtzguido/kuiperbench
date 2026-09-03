import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
from _kuiper_conv2d_general import kuiper_conv2d_general
import torch.nn as nn
import torch.nn.functional as F

class ModelNew(nn.Module):
    """KB L1 #50: AlexNet-style first conv (Conv2d(3, 96, 11, stride=4, pad=2),
    bias=True default)."""
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11,
                               stride=4, padding=2)

    def forward(self, x):
        return kuiper_conv2d_general(x, self.conv1.weight, self.conv1.bias,
                                     stride=4, padding=2,
                                     dilation=1, groups=1)
