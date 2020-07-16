import torch
from torch import nn
from model.vae2 import VAE_Model
from model.discriminator import Discriminator_sketch, Discriminator_face

from edflow import get_logger

class CycleGAN_Model(nn.Module):
    def __init__(self, config):
        super(CycleGAN_Model, self).__init__()
        self.config = config
        self.logger = get_logger("CycleGAN")
        self.output_names = ['real_A', 'fake_B', 'rec_A', 'real_B', 'fake_A', 'rec_B']
        self.cycle = "sketch" in self.config["model_type"] and "face" in self.config["model_type"]
        self.sigma = self.config['variational']['sigma']

        latent_dim = self.config["latent_dim"]
        max_channels= self.config['conv']['n_channel_max']
        min_channels = self.config['conv']['n_channel_start']
        sketch_shape = [32, 1]
        face_shape = [self.config['data']['transform']['resolution'], 3]
        sigma = self.config['variational']['sigma']
        num_extra_conv_sketch = self.config['conv']['sketch_extra_conv']
        num_extra_conv_face = self.config['conv']['face_extra_conv']
        BlockActivation = nn.ReLU()
        FinalActivation = nn.Tanh()
        batch_norm_enc = batch_norm_dec = self.config['batch_norm']
        drop_rate_enc = self.config['dropout']['enc_rate']
        drop_rate_dec = self.config['dropout']['dec_rate']
        bias_enc = self.config['bias']['enc']
        bias_dec = self.config['bias']['dec']
        num_latent_layer = self.config['variational']['num_latent_layer'] if 'num_latent_layer' in self.config['variational'] else 0
        if self.cycle:
            self.netG_A = VAE_Model(latent_dim, min_channels, max_channels, *sketch_shape, *face_shape,
                                    sigma, num_extra_conv_sketch, num_extra_conv_face, 
                                    BlockActivation, FinalActivation,
                                    batch_norm_enc, batch_norm_dec,
                                    drop_rate_enc, drop_rate_dec,
                                    bias_enc, bias_dec, 
                                    num_latent_layer = num_latent_layer)
            self.netD_A = Discriminator_sketch()
            self.netG_B = VAE_Model(latent_dim, min_channels, max_channels, *face_shape, *sketch_shape,
                                    sigma, num_extra_conv_face, num_extra_conv_sketch, 
                                    BlockActivation, FinalActivation,
                                    batch_norm_enc, batch_norm_dec,
                                    drop_rate_enc, drop_rate_dec,
                                    bias_enc, bias_dec, 
                                    num_latent_layer = num_latent_layer)
            self.netD_B = Discriminator_face()
        else:
            sketch = True if "sketch" in self.config["model_type"] else False
            shapes = sketch_shape if "sketch" in self.config["model_type"] else face_shape
            num_extra_conv = num_extra_conv_sketch if "sketch" in self.config["model_type"] else num_extra_conv_face
            self.netG = VAE_Model(latent_dim, min_channels, max_channels, *shapes, *shapes,
                                    sigma, num_extra_conv, num_extra_conv, 
                                    BlockActivation, FinalActivation,
                                    batch_norm_enc, batch_norm_dec,
                                    drop_rate_enc, drop_rate_dec,
                                    bias_enc, bias_dec)
            self.netD = Discriminator_sketch() if sketch else Discriminator_face()

    def vae_forward(self, x):
        return self.netG(x)
        
    def cycle_forward(self, real_A=None, real_B=None):
        '''
        x: dictionary of sketch and face images
        '''
        output_names = []
        output_values = []
        out = []
        #forward pass of images of domain A (sketches)
        if real_A is not None:
            self.logger.debug("first_input.shape " + str(real_A.shape))
            fake_B = self.netG_A(real_A)
            self.logger.debug("first_output secound input .shape " + str(fake_B.shape))
            rec_A = self.netG_B(fake_B)
            self.logger.debug("secound output.shape " + str(rec_A.shape))
            
            output_names += self.output_names[:3]
            output_values +=  [real_A, fake_B, rec_A]
            out.append(fake_B)
        #forward pass of images of domain B (faces)
        if real_B is not None:    
            fake_A = self.netG_B(real_B) 
            self.logger.debug("fake_A.shape " + str(fake_A.shape))
            self.logger.debug("real_B.shape " + str(real_B.shape))
            rec_B = self.netG_A(fake_A)

            output_names += output_names[3:]
            output_values += [real_B, fake_A, rec_B]
            out.append(fake_A)
        
        self.output = dict(zip(self.output_names, output_values))
        return out

    def forward(self, real_A, real_B=None):
        if self.cycle:
            return self.cycle_forward(real_A, real_B)
        else:
            return self.vae_forward(real_A)