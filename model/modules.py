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

class LinLayers(nn.Module):
    def __init__(self, num_lin_layers, latent_dim, sigma):
        super(LinLayers,self).__init__()
        self.num_nodes = latent_dim 
        if sigma:
            self.num_nodes *= 2
        layer_list = []
        for i in range(num_lin_layers):
            layer_list.append(nn.Linear(self.num_nodes, self.num_nodes))
            layer_list.append(nn.BatchNorm1d(self.num_nodes))
            layer_list.append(nn.ReLU())
        self.layers = nn.Sequential(*layer_list)
    
    def forward(self, x):
        change_shape = False
        if len(x.shape) != 2:
            original_shape = x.shape
            change_shape = True
            x = x.reshape(original_shape[0], self.num_nodes)
        x = self.layers(x)
        if change_shape:
            x = x.reshape(*original_shape)
        return x

class Transpose2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, Activation=nn.ReLU(), batch_norm=True, drop_rate=None, bias=False, kernel_size=4, stride=2, padding=1):
        super(Transpose2dBlock, self).__init__()
        layers = [nn.ConvTranspose2d( in_channels, out_channels, kernel_size, stride, padding, bias=bias)]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        if drop_rate is not None:
            layers.append(nn.Dropout(drop_rate))
        layers.append(Activation)

        self.main = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.main(x)


class ExtraConvBlock(nn.Module):
    def __init__(self, channels, Activation=nn.ReLU(), batch_norm=True, drop_rate=None, bias=False):
        super(ExtraConvBlock, self).__init__()
        layers = [nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1)]
        if batch_norm:
            layers.append(nn.BatchNorm2d(channels))
        if drop_rate is not None:
            layers.append(nn.Dropout(drop_rate))
        layers.append(Activation)

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, Activation=nn.ReLU(), batch_norm=True, drop_rate=None, bias=False, kernel_size=3, stride=2, padding=1):
        super(Conv2dBlock, self).__init__()
        layers = [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        if drop_rate is not None:
            layers.append(nn.Dropout(drop_rate))
        layers.append(Activation)

        self.main = nn.Sequential(*layers)
    def forward(self, x):
        return self.main(x)