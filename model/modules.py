# The modules in theis file are copied from the VUNet repository: https://github.com/jhaux/VUNet.git
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F

class NormConv2d(nn.Module):
    """
    Convolutional layer with l2 weight normalization and learned scaling parameters
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.beta = nn.Parameter(
            torch.zeros([1, out_channels, 1, 1], dtype=torch.float32)
        )
        self.gamma = nn.Parameter(
            torch.ones([1, out_channels, 1, 1], dtype=torch.float32)
        )
        self.conv = weight_norm(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            name="weight",
        )

    def forward(self, x):
        # weight normalization
        out = self.conv(x)
        out = self.gamma * out + self.beta
        return out

class Downsample(nn.Module):
    def __init__(self, channels, out_channels=None, kernel_size=3, stride=2, padding=1, conv_layer=NormConv2d, batch_norm = False):
        super().__init__()
        if out_channels == None:
            self.down = conv_layer(
                channels, channels, kernel_size=kernel_size, stride=stride, padding=padding
            )
            self.norm = nn.BatchNorm2d(channels) if batch_norm else ID_module()
        else:
            self.down = conv_layer(
                channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding
            )
            self.norm = nn.BatchNorm2d(out_channels) if batch_norm else ID_module()

    def forward(self, x):
        return self.norm(self.down(x))

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, subpixel=True, conv_layer=NormConv2d, batch_norm = False):
        super().__init__()
        if subpixel:
            self.up = conv_layer(in_channels, 4 * out_channels, 3, padding=1)
            self.op2 = DepthToSpace(block_size=2)
            self.norm = nn.BatchNorm2d(out_channels) if batch_norm else ID_module()
        else:
            # channels have to be bisected because of formely concatenated skips connections
            self.up = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=3, stride=2, padding=1
            )
            self.op2 = ID_module()
            self.norm = nn.BatchNorm2d(out_channels) if batch_norm else ID_module()

    def forward(self, x):
        x = self.up(x)
        x = self.op2(x)
        x = self.norm(x)
        return x

class DepthToSpace(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        n, c, h, w = x.size()
        x = x.view(n, self.bs, self.bs, c // (self.bs ** 2), h, w)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()
        x = x.view(n, c // (self.bs ** 2), h * self.bs, w * self.bs)
        return x

class ID_module(nn.Module):
    def forward(self, input):
        return input

class One_sided_padding(nn.Module):
  def __init__(self, padding = 1):
      super().__init__()
      self.pad = padding
  def forward(self, x):
      return F.pad(x, (0,self.pad,0,self.pad), mode='constant', value=0)