import torch
import torchvision
import torch.nn as nn
from torch.nn import ModuleDict
import torch.nn.functional as F

from edflow import get_logger
from edflow.custom_logging import LogSingleton
import numpy as np

from model.util import (
    get_tensor_shapes,
    get_act_func,
    test_config,
    set_random_state
)
from model.modules import (
    NormConv2d,
    Downsample,
    Upsample
)

class VAE_Model(nn.Module):
    def __init__(self, config):
        super(VAE_Model, self).__init__()
        # set log level to debug if requested
        if "debug_log_level" in config and config["debug_log_level"]:
            LogSingleton.set_log_level("debug")
        # get logger and config
        self.logger = get_logger("VAE_Model")
        # Test the config
        test_config(config)
        self.config = config
        set_random_state(self.config)
        # calculate the tensor shapes throughout the network
        self.tensor_shapes_enc = get_tensor_shapes(config, encoder = True)
        self.tensor_shapes_dec = get_tensor_shapes(config, encoder = False)
        self.logger.info("tensor shapes: " + str(self.tensor_shapes_enc))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # extract information from config
        self.variational = bool("variational" in self.config) 
        self.sigma       = bool(self.variational and "sigma" in self.config["variational"] and self.config["variational"]["sigma"])        
        self.model_type = self.config["model_type"]
        if self.config["model_type"] == "sketch":
            self.enc_extra_conv = self.config["conv"]["sketch_extra_conv"] if "sketch_extra_conv" in self.config["conv"] else 0
            self.dec_extra_conv = self.enc_extra_conv 
        elif self.config["model_type"] == "face":
            self.enc_extra_conv = self.config["conv"]["face_extra_conv"] if "face_extra_conv" in self.config["conv"] else 0
            self.dec_extra_conv = self.enc_extra_conv
        elif self.config["model_type"] == "sketch2face":
            # TODO
            assert 1==0 , "Not ready yet"

        if self.variational:
            if self.sigma:
                self.latent_dim = int(self.tensor_shapes_enc[-1][0]/2)
                self.logger.debug("decoder shapes: " + str(self.tensor_shapes_dec))
            else:
                self.latent_dim = self.tensor_shapes_enc[-1][-1]
        else:
            self.latent_dim = self.config["conv"]["n_channel_max"]
        self.logger.info("latnet dim: " + str(self.latent_dim))
        # how many downsampling blow will we need
        resolution = config["data"]["transform"]["resolution"] if config["model_type"] == "face" else 32
        n_blocks = int(np.round(np.log2(resolution)))
        # get the activation function
        self.act_func = get_act_func(config, self.logger)
        # craete encoder and decoder
        self.enc = VAE_Model_Encoder(config = config, act_func = self.act_func, tensor_shapes = self.tensor_shapes_enc, n_blocks = n_blocks, variaional = self.variational, sigma = self.sigma, latent_dim = self.latent_dim, extra_conv = self.enc_extra_conv)
        self.dec = VAE_Model_Decoder(config = config, act_func = self.act_func, tensor_shapes = self.tensor_shapes_dec, n_blocks = n_blocks, variaional = self.variational, sigma = self.sigma, latent_dim = self.latent_dim, extra_conv = self.dec_extra_conv)

    def direct_z_sample(self, z):
        z = z.to(self.device)
        x = self.dec(z)
        return x

    def latent_sample(self, mu, var = 1, batch_size = None):
        """Sample in the latent space. Input a gaussian distribution to retrive an image from the decoder.
        
        :param mu: The expected value of the gaussian distribution. For only one vaule you can specify a float. For interesting sampling should be a Tensor with dimension same as latent dimension.
        :type mu: Tensor or float
        :param sig: The standard deviation of the gaussian distribution. For only one vaule you can specify a float. For interesting sampling should be a Tensor with dimension same as latent dimension if the model is build for a custom standard deviation, defaults to 1
        :type sig: Tensor/float, optional
        :return: Retruns an image according to the sample.
        :rtype: Tensor
        """        
        assert "variational" in self.config, "If you want to sample from a gaussian distribution to create images you need the key 'variational' in the config."
        if batch_size == None:
            batch_size = self.config["batch_size"]
        if type(mu) in [int,float]:
            mu  = torch.ones([batch_size, self.latent_dim]).to(self.device) * mu
            var = torch.ones([batch_size, self.latent_dim]).to(self.device) * var
        else:
            assert mu.shape[-1] == self.latent_dim, "Wrong shape for latent vector mu"
        if "sigma" not in self.config["variational"] or not self.config["variational"]["sigma"]:
            if var != 1:
                self.logger.info("Variational: sigma is false, var will be overwritten and set to one")
                var = 1
        # create normal distribution 
        norm_dist = torch.distributions.normal.Normal(torch.zeros([batch_size, self.latent_dim]), torch.ones([batch_size, self.latent_dim]))
        eps = norm_dist.sample()
        eps = eps.to(self.device)
        z = mu + var * eps
        z = z.to(self.device)
        x = self.dec(z)
        return x

    def bottleneck(self, x):
        if self.variational:
            norm_dist = torch.distributions.normal.Normal(torch.zeros([x.shape[0], self.latent_dim]), torch.ones([x.shape[0], self.latent_dim]))
            eps = norm_dist.sample().to(self.device)
            if self.sigma:
                self.mu  = x[:, :self.latent_dim]
                self.var = torch.abs(x[:, self.latent_dim:]) + 0.00001
                #self.logger.debug("varitaional mu.shape: " + str(self.mu.shape))
                #self.logger.debug("varitaional var.shape: " + str(self.var.shape))
            else:
                self.mu  = x
                self.var = 1
                #self.logger.debug("varitaional mu.shape: " + str(self.mu.shape))
            # final latent representatione
            x = self.mu + self.var * eps
        return x
        
    def encode_images_to_z(self, x):
        x = self.enc(x)
        self.z = self.bottleneck(x)
        #self.logger.debug("latent size: " + str(self.z.shape))
        return self.z
        

    def forward(self, x):
        """Encodes an image x into the latent represenation z and returns an image generated from that represenation."""        
        x = self.enc(x)
        self.z = self.bottleneck(x)
        #self.logger.debug("output: " + str(self.z.shape))   
        x = self.dec(self.z)
        #self.logger.debug("final output: " + str(x.shape))
        return x

class VAE_Model_Encoder(nn.Module):
    """This is the encoder for the VAE model."""    
    def __init__(
        self, 
        config,
        act_func, 
        tensor_shapes,
        n_blocks, 
        conv       = NormConv2d,
        variaional = None, 
        sigma      = None, 
        latent_dim = None,
        extra_conv = 0
    ):
        super(VAE_Model_Encoder,self).__init__()
        self.logger = get_logger("VAE_Model_Encoder")
        # save all required parameters
        self.config = config 
        self.act_func = act_func
        self.tensor_shapes = tensor_shapes
        self.n_blocks = n_blocks
        self.conv       = conv
        self.variaional = variaional 
        self.sigma      = sigma
        self.latent_dim = latent_dim
        self.extra_conv = extra_conv
        
        self.conv_seq = self.get_module_blocks()

    def get_module_blocks(self):
        # Create the convolutional blocks specified in the config.
        conv_modules_list = []
        for i in range(0, self.n_blocks):
            self.logger.debug("modules: " + str(i))
            for j in range(self.extra_conv):
                channels_out = self.tensor_shapes[i+1][0]
                channels_in = self.tensor_shapes[i][0] if i == 0 and j == 0 else channels_out
                self.logger.debug("extra_conv " + str(j) + "; in_channels " + str(channels_in))
                self.logger.debug("extra_conv " + str(j) + "; out_channels " + str(channels_out))
                conv_modules_list.append(self.conv(in_channels = channels_in, out_channels = channels_out, kernel_size = 3, stride=1, padding=1))
                conv_modules_list.append(self.act_func)
                channels_in = channels_out
            if self.extra_conv == 0:
                channels_in = self.tensor_shapes[i][0]
                channels_out = self.tensor_shapes[i+1][0]
            else:
                channels_out = self.tensor_shapes[i+2][0]
            
            self.logger.debug("in_channels " + str(channels_in))
            self.logger.debug("out_channels " + str(channels_out))
            batch_norm = True if "batch_norm" in self.config and self.config["batch_norm"] else False
            conv_modules_list.append(
            Downsample(channels = channels_in, out_channels = channels_out, kernel_size = 3, stride = 2, padding = 1, conv_layer = self.conv, batch_norm = batch_norm) 
            )    
            conv_modules_list.append(self.act_func)
        return nn.Sequential(*conv_modules_list)
        
    def forward(self, x):
        # Apply all modules in the seqence
        x = self.conv_seq(x)
        self.logger.debug("after all conv blocks x.shape: " + str(x.shape))
        if self.variaional:
            self.logger.debug("latent x.shape: " + str(x.shape))
            self.logger.debug("should be latent shape: [" + str(self.config["batch_size"]) + "," + str(self.tensor_shapes[-1]) + "]")
            x = x.view(-1, self.tensor_shapes[-1][0])
        return x
    
class VAE_Model_Decoder(nn.Module):
    def __init__(
        self,
        config, 
        act_func, 
        tensor_shapes,
        n_blocks,
        conv = NormConv2d,
        variaional = None,
        sigma      = None,
        latent_dim = None,
        extra_conv = 0
    ):
        super(VAE_Model_Decoder,self).__init__()
        self.logger = get_logger("VAE_Model_Decoder")
        # save all required parameters
        self.config = config
        self.act_func = act_func
        self.tensor_shapes = tensor_shapes
        self.n_blocks = n_blocks
        self.conv       = conv
        self.variaional = variaional
        self.sigma      = sigma
        self.latent_dim = latent_dim
        self.extra_conv = extra_conv
        
        self.conv_seq = self.get_module_blocks()
    
    def get_module_blocks(self):
        upsample_modules_list = []
        for i in range(self.n_blocks, 0, -1):
            ba_norm = True if "batch_norm" in self.config and self.config["batch_norm"] and i != 1 else False
            upsample_modules_list.append(Upsample(in_channels = self.tensor_shapes[i][0], out_channels = self.tensor_shapes[i-1][0], conv_layer = self.conv, batch_norm = ba_norm))
            if i != 1:
                upsample_modules_list.append(self.act_func)
            elif self.extra_conv == 0 and i == 1:
                upsample_modules_list.append(nn.Tanh())
            for j in range(self.extra_conv):
                upsample_modules_list.append(self.conv(in_channels = self.tensor_shapes[i-1][0], out_channels = self.tensor_shapes[i-1][0], kernel_size = 3, stride=1, padding=1))
                if i == 1 and j == self.extra_conv-1:
                    upsample_modules_list.append(nn.Tanh())
                else:
                    upsample_modules_list.append(self.act_func)
                    
        return nn.Sequential(*upsample_modules_list)

    def forward(self, x):
        x = x.reshape(-1,*self.tensor_shapes[-1])
        x = self.conv_seq(x)
        return x