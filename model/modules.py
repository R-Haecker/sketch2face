import torch
import torch.nn as nn

class ID_module(nn.Module):
    def forward(self, input):
        return input

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