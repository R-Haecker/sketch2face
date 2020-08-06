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
        self.wasserstein = True if self.config["losses"]['adversarial_loss'] == 'wasserstein' else False

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
        self.netD = Discriminator_sketch(droprate=self.disc_drop_rate, wasserstein=self.wasserstein) if self.sketch else Discriminator_face(droprate=self.disc_drop_rate, wasserstein=self.wasserstein)

    def forward(self, x):
        return self.netG(x)




class WGAN(nn.Module):
    def __init__(self, config):
        super(WGAN, self).__init__()
        
        if "debug_log_level" in config and config["debug_log_level"]:
            LogSingleton.set_log_level("debug")
        # get logger and config
        self.logger = get_logger("VAE_Model")
        self.config = config
        set_random_state(self.config)
        
        self.logger.info("WGAN_GradientPenalty init model.")
        if self.config["model_type"] == "face": self.C = 3
        if self.config["model_type"] == "sketch": self.C = 1

        self.G = WGAN_Generator(self.C)
        self.D = WGAN_Discriminator(input_channels = self.C)

        # WGAN values from paper
        self.learning_rate = 1e-4
        self.b1 = 0.5
        self.b2 = 0.999
        self.batch_size = 64

        self.learning_rate = config["learning_rate"]
        self.batch_size = self.config["batch_size"]

        # WGAN_gradient penalty uses ADAM
        self.d_optimizer = optim.Adam(self.D.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))
        self.g_optimizer = optim.Adam(self.G.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))

        # Set the logger
        self.number_of_images = 10

        self.generator_iters = self.config["num_steps"]
        self.critic_iter = 5
        self.lambda_term = 10

class WGAN_Generator(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Filters [1024, 512, 256]
        # Input_dim = 100
        # Output_dim = C (number of channels)
        self.main_module = nn.Sequential(
            # Z latent vector 100
            nn.ConvTranspose2d(in_channels=100, out_channels=1024, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(True),

            # State (1024x4x4)
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(True),

            # State (512x8x8)
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(True),

            # State (256x16x16)
            nn.ConvTranspose2d(in_channels=256, out_channels=channels, kernel_size=4, stride=2, padding=1))
            # output of main module --> Image (Cx32x32)

        self.output = nn.Tanh()

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)