import torch
import torch.nn as nn
import os
from edflow import get_logger
from edflow.custom_logging import LogSingleton
import numpy as np
import torch.utils.data
from model.modules import Transpose2dBlock, ExtraConvBlock, Conv2dBlock

class VAE_Model(nn.Module):
    def __init__(self,
            latent_dim, 
            min_channels,
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
            same_max_channels=True):
        super(VAE_Model, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = get_logger("VAE_Model")
        self.logger.debug("latent_dim: {}".format(latent_dim))
        self.logger.debug("min_channels: {}".format(min_channels))
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

        self.latent_dim = latent_dim
        self.sigma = sigma

        enc_min_channels = dec_min_channels = min_channels
        if same_max_channels:
            power = np.log2(in_size)
            enc_min_channels = int(2**(power - (np.log2(out_size) - np.log2(in_size)) - 2))
            self.logger.info("Adjusted min enc_min_channels to {} for max_channels of encoder and decoder to match.".format(enc_min_channels))

        self.enc = VAE_Model_Encoder(latent_dim, enc_min_channels, in_size, in_channels, sigma, num_extra_conv_enc, BlockActivation, batch_norm_enc, drop_rate_enc, bias_enc)
        self.dec = VAE_Model_Decoder(latent_dim, dec_min_channels, out_size, out_channels, num_extra_conv_dec, BlockActivation, FinalActivation, batch_norm_dec, drop_rate_dec, bias_dec)


    def bottleneck(self, x):
        norm_dist = torch.distributions.normal.Normal(torch.zeros([x.shape[0], self.latent_dim]), torch.ones([x.shape[0], self.latent_dim]))
        eps = norm_dist.sample().to(self.device)
        if self.sigma:
            mu  = x[:, :self.latent_dim]
            var = torch.abs(x[:, self.latent_dim:]) + 0.00001
            self.logger.debug("varitaional mu.shape: " + str(mu.shape))
            self.logger.debug("varitaional var.shape: " + str(var.shape))
        else:
            mu  = x
            var = 1
            self.logger.debug("varitaional mu.shape: " + str(mu.shape))
        # final latent representatione
        x = mu + var * eps
        return x

    def forward(self, x):
        x = self.enc(x)
        x = x.reshape(x.shape[:-2])
        x = self.bottleneck(x)
        x = self.dec(x)
        return x


class VAE_Model_Encoder(nn.Module):
    def __init__(self,
        latent_dim,
        min_channels,
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
        channel_numbers = [in_channels] + list(min_channels * 2**np.arange(np.log2(in_size)).astype(np.int)) + [latent_dim]
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
        channel_numbers = [latent_dim] + list(min_channels * 2**np.arange(np.log2(out_size)-2)[::-1].astype(np.int))
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
