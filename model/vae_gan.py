from torch import nn
from model.vae import VAE_config
from model.discriminator import Discriminator_sketch, Discriminator_face, WGAN_Discriminator_sketch, WGAN_Discriminator_face
from model.util import set_random_state

from edflow.custom_logging import LogSingleton
from edflow import get_logger

#######################
###  VAE GAN Model  ###
#######################

class VAE_GAN(nn.Module):
    def __init__(self, config):
        super(VAE_GAN, self).__init__()
        self.config = config
        if "debug_log_level" in config and config["debug_log_level"]:
            LogSingleton.set_log_level("debug")
        self.logger = get_logger("VAE_GAN")
        assert bool("sketch" in self.config["model_type"]) != bool("face" in self.config["model_type"]), "The model_type for this VAE GAN model can only be 'sketch' or 'face' but not 'sketch2face'."
        assert config["iterator"] == "iterator.vae_gan.VAE_GAN", "This model supports only the VAE_GAN iterator."
        set_random_state(self.config)
        self.sigma = self.config["variational"]["sigma"] if "variational" in self.config and "sigma" in self.config["variational"] else False
        sketch = True if "sketch" in self.config["model_type"] else False
        self.netG = VAE_config(self.config)
        self.netD = Discriminator_sketch() if sketch else Discriminator_face()

    def forward(self, x):
        return self.netG(x)

########################
###  VAE WGAN Model  ###
########################

class VAE_WGAN(nn.Module):
    def __init__(self, config):
        super(VAE_WGAN, self).__init__()
        if "debug_log_level" in config and config["debug_log_level"]:
            LogSingleton.set_log_level("debug")
        # get logger and config
        self.logger = get_logger("CycleWGAN")
        self.config = config
        set_random_state(self.config)
        assert bool("sketch" in self.config["model_type"]) != bool("face" in self.config["model_type"]), "The model_type for this VAE GAN model can only be 'sketch' or 'face' but not 'sketch2face'."
        assert config["iterator"] == "iterator.vae_wgan.VAE_WGAN", "This model supports only the VAE_WGAN iterator."
        self.logger.info("VAE WGAN init model.")
        self.sigma = self.config['variational']['sigma'] if "variational" in self.config and "sigma" in self.config["variational"] else False
        
        self.netG = VAE_config(self.config)
        self.netD = WGAN_Discriminator_sketch() if "sketch" in self.config["model_type"] else WGAN_Discriminator_face(input_resolution=config["data"]["transform"]["resolution"])

    def forward(self, x):
        return self.netG(x)