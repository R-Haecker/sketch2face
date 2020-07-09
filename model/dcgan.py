import torch
from torch import nn
from model.vae import VAE_Model_Decoder
from model.discriminator import Discriminator_sketch, Discriminator_face

from edflow import get_logger

from model.util import (
    get_tensor_shapes,
    get_act_func,
    test_config,
    set_random_state
)

class DCGAN_Model(nn.Module):
    def __init__(self, config):
        super(DCGAN_Model, self).__init__()
        self.config = config
        self.logger = get_logger("DCGAN")
        self.sketch = True if "sketch" in self.config["model_type"] else False
        self.tensor_shapes_enc = get_tensor_shapes(config, sketch = self.sketch, encoder = True)
        self.tensor_shapes_dec = get_tensor_shapes(config, sketch = self.sketch, encoder = False)
        self.logger.info("tensor shapes decoder input: " + str(self.tensor_shapes_dec))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.variational = bool("variational" in self.config) 
        self.sigma       = bool(self.variational and "sigma" in self.config["variational"] and self.config["variational"]["sigma"])    
        if "dropout" in self.config:
            self.dec_drop_rate = self.config["dropout"]['dec_rate'] if 'dec_rate' in self.config["dropout"] else None
            self.disc_drop_rate = self.config["dropout"]['disc_rate'] if ('disc_rate' in self.config["dropout"]) else 0  
            self.logger.debug("Dropout oused in discriminator with disc_rate = {}".format(self.disc_drop_rate))

        key_dec = "sketch_extra_conv" if self.sketch else "face_extra_conv"
        self.dec_extra_conv = self.config["conv"][key_dec] if key_dec in self.config["conv"] else 0
        
        if self.variational:
            if self.sigma:
                self.latent_dim = int(self.tensor_shapes_enc[-1][0]/2)
                self.logger.debug("decoder shapes: " + str(self.tensor_shapes_dec))
            else:
                self.latent_dim = self.tensor_shapes_enc[-1][0]
        else:
            self.latent_dim = self.config["conv"]["n_channel_max"]
        self.logger.info("latnet dim: " + str(self.latent_dim))

        self.act_func = get_act_func(config, self.logger)
        dec_n_blocks = len(self.tensor_shapes_dec)-1 if self.dec_extra_conv == 0 else len(self.tensor_shapes_dec)-2 

        self.netG = VAE_Model_Decoder(config = config, act_func = self.act_func, tensor_shapes = self.tensor_shapes_dec, n_blocks = dec_n_blocks, variaional = self.variational, sigma = self.sigma, latent_dim = self.latent_dim, extra_conv = self.dec_extra_conv, drop_rate = self.dec_drop_rate)
        self.netD = Discriminator_sketch(droprate=self.disc_drop_rate) if self.sketch else Discriminator_face(droprate=self.disc_drop_rate)

    def forward(self, x):
        return self.netG(x)