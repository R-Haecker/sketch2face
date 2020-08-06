from torch import nn
from model.vae import VAE_config
from model.discriminator import Discriminator_sketch, Discriminator_face

from edflow import get_logger

#######################
###  VAE GAN Model  ###
#######################

class VAE_GAN(nn.Module):
    def __init__(self, config):
        super(VAE_GAN, self).__init__()
        self.config = config
        self.logger = get_logger("VAE_GAN")
        assert bool("sketch" in self.config["model_type"]) != bool("face" in self.config["model_type"]), "The model_type for this VAE GAN model can only be 'sketch' or 'face' but not 'sketch2face'."
        sketch = True if "sketch" in self.config["model_type"] else False
        self.netG = VAE_config(self.config)
        self.netD = Discriminator_sketch() if sketch else Discriminator_face()

    def forward(self, x):
        return self.netG(x)