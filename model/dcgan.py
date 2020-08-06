import torch
from torch import nn
import torch.optim as optim
from model.vae import VAE_Decoder
from model.discriminator import Discriminator_sketch, Discriminator_face, WGAN_Discriminator

from edflow import get_logger

from model.util import (
    get_tensor_shapes,
    get_act_func,
    test_config,
    set_random_state
)

class DCGAN(nn.Module):
    def __init__(self, config):
        super(DCGAN, self).__init__()
        self.config = config
        self.logger = get_logger("DCGAN")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert bool("sketch" in self.config["model_type"]) != bool("face" in self.config["model_type"]), "The model_type for this DCGAN model can only be 'sketch' or 'face' but not 'sketch2face'."
        self.sketch = True if "sketch" in self.config["model_type"] else False
        self.wasserstein = True if self.config["losses"]['adversarial_loss'] == 'wasserstein' else False

        latent_dim = self.config['conv']['n_channel_max']
        min_channels = self.config['conv']['n_channel_start']
        sketch_shape = [32, 1]
        face_shape = [self.config['data']['transform']['resolution'], 3]
        num_extra_conv_sketch = self.config['conv']['sketch_extra_conv']
        num_extra_conv_face = self.config['conv']['face_extra_conv']
        BlockActivation = nn.ReLU()
        FinalActivation = nn.Tanh()
        batch_norm_dec = self.config['batch_norm']
        drop_rate_dec = self.config['dropout']['dec_rate']
        drop_rate_disc = self.config['dropout']['disc_rate']
        bias_dec = self.config['bias']['dec']

        shapes = sketch_shape if "sketch" in self.config["model_type"] else face_shape
        num_extra_conv = num_extra_conv_sketch if "sketch" in self.config["model_type"] else num_extra_conv_face
        self.netG = VAE_Decoder(latent_dim, min_channels, 
                                    *shapes, num_extra_conv, 
                                    BlockActivation, FinalActivation,
                                    batch_norm_dec, drop_rate_dec, bias_dec)
        
        self.netD = Discriminator_sketch(droprate=drop_rate_disc, wasserstein=self.wasserstein) if self.sketch else Discriminator_face(droprate=drop_rate_disc, wasserstein=self.wasserstein)

    def forward(self, x):
        return self.netG(x)