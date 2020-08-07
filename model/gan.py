import torch
from torch import nn
import torch.optim as optim
from model.vae import VAE_Decoder
from model.discriminator import Discriminator_sketch, Discriminator_face, WGAN_Discriminator
from model.util import set_random_state

from edflow.custom_logging import LogSingleton
from edflow import get_logger

class DCGAN(nn.Module):
    def __init__(self, config):
        super(DCGAN, self).__init__()
        if "debug_log_level" in config and config["debug_log_level"]:
            LogSingleton.set_log_level("debug")
        self.config = config
        self.logger = get_logger("DCGAN")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert bool("sketch" in self.config["model_type"]) != bool("face" in self.config["model_type"]), "The model_type for this DCGAN model can only be 'sketch' or 'face' but not 'sketch2face'."
        self.sketch = True if "sketch" in self.config["model_type"] else False
        self.wasserstein = bool(self.config["losses"]['adversarial_loss'] == 'wasserstein')

        latent_dim = self.config['latent_dim']
        min_channels = self.config['conv']['n_channel_start']
        max_channels = self.config['conv']['n_channel_max']
        sketch_shape = [32, 1]
        face_shape = [self.config['data']['transform']['resolution'], 3]
        num_extra_conv_sketch = self.config['conv']['sketch_extra_conv']
        num_extra_conv_face = self.config['conv']['face_extra_conv']
        block_activation = nn.ReLU()
        final_activation = nn.Tanh()
        batch_norm_dec = self.config['batch_norm']
        drop_rate_dec = self.config['dropout']['dec_rate']
        drop_rate_disc = self.config['dropout']['disc_rate']
        bias_dec = self.config['bias']['dec']

        shapes = sketch_shape if "sketch" in self.config["model_type"] else face_shape
        num_extra_conv = num_extra_conv_sketch if "sketch" in self.config["model_type"] else num_extra_conv_face
        self.netG = VAE_Decoder(latent_dim, min_channels, max_channels,
                                *shapes, num_extra_conv, 
                                block_activation, final_activation,
                                batch_norm_dec, drop_rate_dec, bias_dec)
        self.netD = Discriminator_sketch(droprate=drop_rate_disc, wasserstein=self.wasserstein) if self.sketch else Discriminator_face(droprate=drop_rate_disc, wasserstein=self.wasserstein)

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

        self.netG = WGAN_Generator(self.C)
        self.netD = WGAN_Discriminator(input_channels = self.C)

        # WGAN values from paper
        self.b1 = 0.5
        self.b2 = 0.999

        self.learning_rate = config["learning_rate"]
        self.batch_size = self.config["batch_size"]

        # WGAN_gradient penalty uses ADAM
        self.d_optimizer = optim.Adam(self.D.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))
        self.g_optimizer = optim.Adam(self.G.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))

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