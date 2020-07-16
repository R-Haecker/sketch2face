import torch
import torch.nn as nn
import os
from edflow import get_logger
from edflow.custom_logging import LogSingleton
import numpy as np
import torch.utils.data
from model.modules import Transpose2dBlock, ExtraConvBlock, Conv2dBlock, LinLayers
from torch.autograd import Variable

class VAE_Model(nn.Module):
    def __init__(self,
            latent_dim, 
            min_channels,
            max_channels,
            in_size, 
            in_channels, 
            out_size,
            out_channels,
            sigma=False,
            num_extra_conv_enc=0,
            num_extra_conv_dec=0,
            BlockActivation=nn.ReLU(),
            FinalActivation=nn.Tanh(),
            batch_norm_enc=True,
            batch_norm_dec=True,
            drop_rate_enc=None,
            drop_rate_dec=None,
            bias_enc=False,
            bias_dec=False,
            same_max_channels=False,
            num_latent_layer=0):
        super(VAE_Model, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = get_logger("VAE_Model")
        self.logger.debug("latent_dim: {}".format(latent_dim))
        self.logger.debug("min_channels: {}".format(min_channels))
        self.logger.debug("max_channels: {}".format(max_channels))
        self.logger.debug("in_size: {}".format(in_size))
        self.logger.debug("in_channels: {}".format(in_channels))
        self.logger.debug("out_size: {}".format(out_size))
        self.logger.debug("out_channels: {}".format(out_channels))
        self.logger.debug("sigma: {}".format(sigma))
        self.logger.debug("num_extra_conv_enc: {}".format(num_extra_conv_enc))
        self.logger.debug("num_extra_conv_dec: {}".format(num_extra_conv_dec))
        self.logger.debug("BlockActivation: {}".format(BlockActivation))
        self.logger.debug("FinalActivation: {}".format(FinalActivation))
        self.logger.debug("batch_norm_enc: {}".format(batch_norm_enc))
        self.logger.debug("batch_norm_dec: {}".format(batch_norm_dec))
        self.logger.debug("drop_rate_enc: {}".format(drop_rate_enc))
        self.logger.debug("drop_rate_dec: {}".format(drop_rate_dec))
        self.logger.debug("bias_enc: {}".format(bias_enc))
        self.logger.debug("bias_dec: {}".format(bias_dec))
        self.logger.debug("same_max_channels: {}".format(same_max_channels))
        self.logger.debug("num_latent_layer: {}".format(num_latent_layer))

        self.latent_dim = latent_dim
        self.sigma = sigma

        enc_min_channels = dec_min_channels = min_channels
        enc_max_channels = dec_max_channels = max_channels
        if same_max_channels:
            power = np.log2(in_size)
            enc_min_channels = int(2**(power - (np.log2(out_size) - np.log2(in_size)) - 2))
            self.logger.info("Adjusted min enc_min_channels to {} for max_channels of encoder and decoder to match.".format(enc_min_channels))

        self.enc = VAE_Model_Encoder(latent_dim, enc_min_channels, enc_max_channels, in_size, in_channels, sigma, num_extra_conv_enc, BlockActivation, batch_norm_enc, drop_rate_enc, bias_enc)
        self.dec = VAE_Model_Decoder(latent_dim, dec_min_channels, dec_max_channels, out_size, out_channels, num_extra_conv_dec, BlockActivation, FinalActivation, batch_norm_dec, drop_rate_dec, bias_dec)

        self.add_linear_layers = False
        if num_latent_layer > 0:
                self.add_linear_layers = True
                self.latent_layer = LinLayers(num_latent_layer, self.latent_dim, self.sigma)
                self.logger.info("Added {} linear layers layers".format(num_latent_layer))


    def bottleneck(self, x):
        if self.sigma:
            self.mu = x[:, :self.latent_dim]
            self.logvar = x[:, self.latent_dim:]
            self.std = self.logvar.mul(0.5).exp_()
            self.logger.debug("varitaional mu.shape: " + str(self.mu.shape))
            self.logger.debug("varitaional var.shape: " + str(self.std.shape))
        else:
            self.mu  = x
            self.std = torch.ones_like(x)
            self.logger.debug("varitaional mu.shape: " + str(self.mu.shape))
        eps = Variable(self.std.data.new(self.std.size()).normal_())
        # final latent representatione
        x = self.mu + self.std * eps
        return x

    def forward(self, x):
        x = self.enc(x)
        if self.add_linear_layers:
            x = self.latent_layer(x)
        x = x.reshape(x.shape[:-2])
        x = self.bottleneck(x)
        x = self.dec(x)
        return x


class VAE_Model_Encoder(nn.Module):
    def __init__(self,
        latent_dim,
        min_channels,
        max_channels,
        in_size,
        in_channels,
        sigma=False,
        num_extra_conv=0,
        BlockActivation=nn.ReLU(), 
        batch_norm=True, 
        drop_rate=None,
        bias=False):

        super(VAE_Model_Encoder, self).__init__()
        self.logger = get_logger("Encoder")

        layers = []
        latent_dim = 2*latent_dim if sigma else latent_dim
        
        print("np.log2(in_size)",np.log2(in_size))
        print("np.ones(np.log2(in_size), dtype=int) * int(max_channels)",np.ones(int(np.log2(in_size)), dtype=int) * int(max_channels))
        print("min_channels * 2**np.arange(np.log2(in_size)).astype(np.int)",min_channels * 2**np.arange(np.log2(in_size)).astype(np.int))
        channel_numbers = [in_channels] + list( np.minimum( min_channels * 2**np.arange(np.log2(in_size)).astype(np.int), np.ones(int(np.log2(in_size)), dtype=int) * int(max_channels) )) + [latent_dim]
        for i in range(len(channel_numbers)-1):
            in_ch = channel_numbers[i]
            out_ch = channel_numbers[i+1]
            for i in range(num_extra_conv):
                layers.append(ExtraConvBlock(in_ch, BlockActivation, batch_norm, drop_rate, bias))
            if i != len(channel_numbers)-2:
                layers.append(Conv2dBlock(in_ch, out_ch, BlockActivation, batch_norm, drop_rate, bias))
            else:
                layers.append(Conv2dBlock(in_ch, out_ch, BlockActivation, batch_norm, drop_rate, bias, stride=1))
        self.main = nn.Sequential(*layers)

        self.logger.debug("Encoder channel sizes: {}".format(channel_numbers))

    def forward(self, x):
        return self.main(x)

class VAE_Model_Decoder(nn.Module):
    def __init__(self,
        latent_dim,
        min_channels,
        max_channels,
        out_size,
        out_channels,
        num_extra_conv=0,
        BlockActivation=nn.ReLU(), 
        FinalActivation=nn.Tanh(),
        batch_norm=True, 
        drop_rate=None,
        bias=False):

        super(VAE_Model_Decoder, self).__init__()
        self.logger = get_logger("Decoder")

        layers = []
        channel_numbers = [latent_dim] + list(np.minimum( min_channels * 2**np.arange(np.log2(out_size)-2)[::-1].astype(np.int), np.ones( int(np.log2(out_size)-2), dtype=int) * int(max_channels)) )
        for i in range(len(channel_numbers)-1):
            in_ch = channel_numbers[i]
            out_ch = channel_numbers[i+1]
            stride = 2 if i != 0 else 1
            padding = 1 if i != 0 else 0
            layers.append(Transpose2dBlock(in_ch, out_ch, BlockActivation, batch_norm, drop_rate, bias, stride=stride, padding=padding))
            for i in range(num_extra_conv):
                layers.append(ExtraConvBlock(out_ch, BlockActivation, batch_norm, drop_rate, bias))
        
        if num_extra_conv > 0:
            layers.append(Transpose2dBlock(min_channels, out_channels, BlockActivation,  batch_norm, drop_rate, bias))
        else:
            layers.append(Transpose2dBlock(min_channels, out_channels, FinalActivation, batch_norm=False, drop_rate=None, bias=bias))
        for i in range(num_extra_conv):
            if i != num_extra_conv-1:
                layers.append(ExtraConvBlock(out_channels, BlockActivation, batch_norm, drop_rate, bias))
            else:
                layers.append(ExtraConvBlock(out_channels, FinalActivation, batch_norm=False, drop_rate=None, bias=bias))
        
        self.main = nn.Sequential(*layers)

        self.logger.debug("Decoder channel sizes: {}".format(channel_numbers + [out_channels]))
    
    def forward(self, x):
        x = x[...,None,None]
        return self.main(x)
