import torch
from torch import nn
from model.vae import VAE
from model.discriminator import Discriminator_sketch, Discriminator_face, WGAN_Discriminator_sketch, WGAN_Discriminator_face
from model.util import set_random_state

from edflow.custom_logging import LogSingleton
from edflow import get_logger

#########################
###  Cycle GAN Model  ###
#########################

class Cycle_GAN(nn.Module):
    def __init__(self, config):
        super(Cycle_GAN, self).__init__()
        self.config = config
        self.logger = get_logger("CycleGAN")
        self.output_names = ['real_A', 'fake_B', 'rec_A', 'real_B', 'fake_A', 'rec_B']
        self.sigma = self.config['variational']['sigma']
        assert "sketch" in self.config["model_type"] and "face" in self.config["model_type"], "This CycleGAN iterator only works with model_type:'sketch2face'"
        assert config["iterator"] == "iterator.cycle_gan.CycleGAN", "This CycleGAN model only works with with the Cycle_GAN iterator."
        
        latent_dim = self.config["latent_dim"]
        min_channels = self.config['conv']['n_channel_start']
        max_channels= self.config['conv']['n_channel_max']
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
        # load the right networks of the model
        self.netG_A = VAE(latent_dim, min_channels, max_channels, *sketch_shape, *face_shape,
                                sigma, num_extra_conv_sketch, num_extra_conv_face, 
                                BlockActivation, FinalActivation,
                                batch_norm_enc, batch_norm_dec,
                                drop_rate_enc, drop_rate_dec,
                                bias_enc, bias_dec, 
                                num_latent_layer = num_latent_layer)
        self.netD_A = Discriminator_sketch()
        self.netG_B = VAE(latent_dim, min_channels, max_channels, *face_shape, *sketch_shape,
                                sigma, num_extra_conv_face, num_extra_conv_sketch, 
                                BlockActivation, FinalActivation,
                                batch_norm_enc, batch_norm_dec,
                                drop_rate_enc, drop_rate_dec,
                                bias_enc, bias_dec, 
                                num_latent_layer = num_latent_layer)
        self.netD_B = Discriminator_face()
        
    def forward(self, real_A=None, real_B=None):
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

##########################
###  Cycle WGAN Model  ###
##########################

class Cycle_WGAN(nn.Module):
    def __init__(self, config):
        super(Cycle_WGAN, self).__init__()
        assert config["iterator"] == "iterator.cycle_wgan.Cycle_WGAN", "This model only works with the iterator: 'iterator.cycle_wgan.Cycle_WGAN"
        assert "sketch" in config["model_type"] and "face" in config["model_type"], "This model only works with model_type: 'sketch2face'"
        if "debug_log_level" in config and config["debug_log_level"]:
            LogSingleton.set_log_level("debug")
        # get logger and config
        self.logger = get_logger("CycleWGAN")
        self.config = config
        set_random_state(self.config)
        self.logger.info("WGAN_GradientPenalty init model.")
        self.output_names = ['real_A', 'fake_B', 'rec_A', 'real_B', 'fake_A', 'rec_B']
        self.sigma = self.config['variational']['sigma']
        self.num_latent_layer = self.config['variational']['num_latent_layer'] if "variational" in self.config and "num_latent_layer" in self.config["variational"] else 0

        latent_dim = self.config["latent_dim"]
        min_channels = self.config['conv']['n_channel_start']
        max_channels= self.config['conv']['n_channel_max']
        sketch_shape = [32, 1]
        face_shape = [self.config['data']['transform']['resolution'], 3]
        sigma = self.config['variational']['sigma']
        num_extra_conv_sketch = self.config['conv']['sketch_extra_conv']
        num_extra_conv_face = self.config['conv']['face_extra_conv']
        block_activation = nn.ReLU()
        final_activation = nn.Tanh()
        batch_norm_enc = batch_norm_dec = self.config['batch_norm']
        drop_rate_enc = self.config['dropout']['enc_rate'] if "dropout" in self.config and "enc_rate" in self.config["dropout"] else 0
        drop_rate_dec = self.config['dropout']['dec_rate'] if "dropout" in self.config and "dec_rate" in self.config["dropout"] else 0
        bias_enc = self.config['bias']['enc'] if "bias" in self.config and "enc" in self.config["bias"] else True
        bias_dec = self.config['bias']['dec'] if "bias" in self.config and "dec" in self.config["bias"] else True
        num_latent_layer = self.config['variational']['num_latent_layer'] if "variational" in self.config and "num_latent_layer" in self.config["variational"] else 0
        ## cycle A ##
        self.netG_A = VAE(
            latent_dim = latent_dim,
            min_channels = min_channels,
            max_channels = max_channels,
            in_size = sketch_shape[0], 
            in_channels = sketch_shape[1],
            out_size = face_shape[0],
            out_channels = face_shape[1],
            sigma=sigma,
            num_extra_conv_enc=num_extra_conv_sketch,
            num_extra_conv_dec=num_extra_conv_face,
            block_activation=block_activation,
            final_activation=final_activation,
            batch_norm_enc=batch_norm_enc,
            batch_norm_dec=batch_norm_dec,
            drop_rate_enc=drop_rate_enc,
            drop_rate_dec=drop_rate_dec,
            bias_enc=bias_enc,
            bias_dec=bias_dec,
            same_max_channels=False, 
            num_latent_layer=num_latent_layer)
        self.netD_A = WGAN_Discriminator_sketch()
        ## cycle B ##
        self.netG_B = VAE(
            latent_dim = latent_dim,
            min_channels = min_channels,
            max_channels = max_channels,
            in_size = face_shape[0], 
            in_channels = face_shape[1],
            out_size = sketch_shape[0],
            out_channels = sketch_shape[1],
            sigma=sigma,
            num_extra_conv_enc=num_extra_conv_sketch,
            num_extra_conv_dec=num_extra_conv_face,
            block_activation=block_activation,
            final_activation=final_activation,
            batch_norm_enc=batch_norm_enc,
            batch_norm_dec=batch_norm_dec,
            drop_rate_enc=drop_rate_enc,
            drop_rate_dec=drop_rate_dec,
            bias_enc=bias_enc,
            bias_dec=bias_dec,
            same_max_channels=False, 
            num_latent_layer=num_latent_layer)
        self.netD_B = WGAN_Discriminator_face(input_resolution=face_shape[0])
        
    def forward(self, real_A=None, real_B=None):
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