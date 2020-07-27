import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch import autograd
import time as t
import matplotlib.pyplot as plt

#plt.switch_backend('agg')

import os
from itertools import chain
from torchvision import utils

from edflow import get_logger
from edflow.custom_logging import LogSingleton
import numpy as np

from model.vae2 import VAE_Model

from model.util import (
    get_tensor_shapes,
    get_act_func,
    test_config,
    set_random_state
)

class Old_Generator(nn.Module):
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

class Discriminator(torch.nn.Module):
    def __init__(self, input_channels, input_resolution):
        super().__init__()
        # Filters [256, 512, 1024]
        # Input_dim = channels (Cx64x64)
        # Output_dim = 1
        
        # Omitting batch normalization in critic because our new penalized training objective (WGAN with gradient penalty) is no longer valid
        # in this setting, since we penalize the norm of the critic's gradient with respect to each input independently and not the enitre batch.
        # There is not good & fast implementation of layer normalization --> using per instance normalization nn.InstanceNorm2d()
        modules = []
        if input_resolution != 32:
            dif = int(np.log2(input_resolution) - np.log2(32))
            assert input_resolution > 32, "input_resolution has to be equal or bigger than 32."
            print("Adding " + str(dif) + " extra layers to discriminator")
            for i in range(dif):
                modules.append(nn.Conv2d(in_channels=input_channels, out_channels=256, kernel_size=4, stride=2, padding=1))
                modules.append(nn.InstanceNorm2d(256, affine=True))
                modules.append(nn.LeakyReLU(0.2, inplace=True))
                input_channels = 256
                
        # Image input (Cx32x32)
        modules.append(nn.Conv2d(in_channels=input_channels, out_channels=256, kernel_size=4, stride=2, padding=1))
        modules.append(nn.InstanceNorm2d(256, affine=True))
        modules.append(nn.LeakyReLU(0.2, inplace=True))
        
        # State (256x16x16)
        modules.append(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1))
        modules.append(nn.InstanceNorm2d(512, affine=True))
        modules.append(nn.LeakyReLU(0.2, inplace=True))
        
        # State (512x8x8)
        modules.append(nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1))
        modules.append(nn.InstanceNorm2d(1024, affine=True))
        modules.append(nn.LeakyReLU(0.2, inplace=True))
        # output of main module --> State (1024x4x4)
        self.main_module = nn.Sequential(*modules)            

        self.output = nn.Sequential(
            # The output of D is no longer a probability, we do not apply sigmoid at the output of D.
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0))


    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)

    def feature_extraction(self, x):
        # Use discriminator for feature extraction then flatten to vector of 16384
        x = self.main_module(x)
        return x.view(-1, 1024*4*4)

class Discriminator_sketch(Discriminator):
    def __init__(self):
        super().__init__(input_channels = 1, input_resolution = 32)

class Discriminator_face(Discriminator):
    def __init__(self, input_resolution):
        super().__init__(input_channels = 3, input_resolution = input_resolution)


class WGAN_GP(nn.Module):
    def __init__(self, config):
        super(WGAN_GP, self).__init__()
        
        if "debug_log_level" in config and config["debug_log_level"]:
            LogSingleton.set_log_level("debug")
        # get logger and config
        self.logger = get_logger("VAE_Model")
        self.config = config
        set_random_state(self.config)
        
        self.logger.info("WGAN_GradientPenalty init model.")
        if self.config["model_type"] == "face": self.C = 3
        if self.config["model_type"] == "sketch": self.C = 1

        self.G = Generator(self.C)
        self.D = Discriminator(input_channels = self.C)

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

    def real_images(self, images, number_of_images):
        if (self.C == 3):
            return self.to_np(images.view(-1, self.C, 32, 32)[:self.number_of_images])
        else:
            return self.to_np(images.view(-1, 32, 32)[:self.number_of_images])

    def generate_img(self, z, number_of_images):
        samples = self.G(z).data.cpu().numpy()[:number_of_images]
        generated_images = []
        for sample in samples:
            if self.C == 3:
                generated_images.append(sample.reshape(self.C, 32, 32))
            else:
                generated_images.append(sample.reshape(32, 32))
        return generated_images

    def to_np(self, x):
        return x.data.cpu().numpy()

    def generate_latent_walk(self, number):
        if not os.path.exists('interpolated_images/'):
            os.makedirs('interpolated_images/')

        number_int = 10
        # interpolate between twe noise(z1, z2).
        z_intp = torch.FloatTensor(1, 100, 1, 1)
        z1 = torch.randn(1, 100, 1, 1)
        z2 = torch.randn(1, 100, 1, 1)
        if self.cuda:
            z_intp = z_intp.cuda()
            z1 = z1.cuda()
            z2 = z2.cuda()

        z_intp = Variable(z_intp)
        images = []
        alpha = 1.0 / float(number_int + 1)
        print(alpha)
        for i in range(1, number_int + 1):
            z_intp.data = z1*alpha + z2*(1.0 - alpha)
            alpha += alpha
            fake_im = self.G(z_intp)
            fake_im = fake_im.mul(0.5).add(0.5) #denormalize
            images.append(fake_im.view(self.C,32,32).data.cpu())

        grid = utils.make_grid(images, nrow=number_int )
        utils.save_image(grid, 'interpolated_images/interpolated_{}.png'.format(str(number).zfill(3)))
        print("Saved interpolated images.")


class CycleWGAN_GP_VAE(nn.Module):
    def __init__(self, config):
        super(CycleWGAN_GP_VAE, self).__init__()
        
        if "debug_log_level" in config and config["debug_log_level"]:
            LogSingleton.set_log_level("debug")
        # get logger and config
        self.logger = get_logger("CycleWGAN")
        self.config = config
        set_random_state(self.config)
        self.logger.info("WGAN_GradientPenalty init model.")
        self.output_names = ['real_A', 'fake_B', 'rec_A', 'real_B', 'fake_A', 'rec_B']
        self.cycle = "sketch" in self.config["model_type"] and "face" in self.config["model_type"]
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
        BlockActivation = nn.ReLU()
        FinalActivation = nn.Tanh()
        batch_norm_enc = batch_norm_dec = self.config['batch_norm']
        drop_rate_enc = self.config['dropout']['enc_rate'] if "dropout" in self.config and "enc_rate" in self.config["dropout"] else 0
        drop_rate_dec = self.config['dropout']['dec_rate'] if "dropout" in self.config and "dec_rate" in self.config["dropout"] else 0
        bias_enc = self.config['bias']['enc'] if "bias" in self.config and "enc" in self.config["bias"] else True
        bias_dec = self.config['bias']['dec'] if "bias" in self.config and "dec" in self.config["bias"] else True
        num_latent_layer = self.config['variational']['num_latent_layer'] if "variational" in self.config and "num_latent_layer" in self.config["variational"] else 0
        
        if self.cycle:
            ## cycle A ##
            self.netG_A = VAE_Model(
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
                BlockActivation=BlockActivation,
                FinalActivation=FinalActivation,
                batch_norm_enc=batch_norm_enc,
                batch_norm_dec=batch_norm_dec,
                drop_rate_enc=drop_rate_enc,
                drop_rate_dec=drop_rate_dec,
                bias_enc=bias_enc,
                bias_dec=bias_dec,
                same_max_channels=False, 
                num_latent_layer=num_latent_layer)
            self.netD_A = Discriminator_sketch()
            ## cycle B ##
            self.netG_B = VAE_Model(
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
                BlockActivation=BlockActivation,
                FinalActivation=FinalActivation,
                batch_norm_enc=batch_norm_enc,
                batch_norm_dec=batch_norm_dec,
                drop_rate_enc=drop_rate_enc,
                drop_rate_dec=drop_rate_dec,
                bias_enc=bias_enc,
                bias_dec=bias_dec,
                same_max_channels=False, 
                num_latent_layer=num_latent_layer)
            self.netD_B = Discriminator_face(input_resolution=face_shape[0])
            self.forward = self.cycle_forward
            self.restore = self.cycle_restore
            self.save = self.cycle_save
        else:
            shapes = sketch_shape if "sketch" in self.config["model_type"] else face_shape
            num_extra_conv = num_extra_conv_sketch if "sketch" in self.config["model_type"] else num_extra_conv_face
            # initialise model with right arguments
            self.netG = VAE_Model(
                latent_dim = latent_dim,
                min_channels = min_channels,
                max_channels = max_channels,
                in_size = shapes[0], 
                in_channels = shapes[1],
                out_size = shapes[0],
                out_channels = shapes[1],
                sigma=sigma,
                num_extra_conv_enc=num_extra_conv,
                num_extra_conv_dec=num_extra_conv,
                BlockActivation=BlockActivation,
                FinalActivation=FinalActivation,
                batch_norm_enc=batch_norm_enc,
                batch_norm_dec=batch_norm_dec,
                drop_rate_enc=drop_rate_enc,
                drop_rate_dec=drop_rate_dec,
                bias_enc=bias_enc,
                bias_dec=bias_dec,
                same_max_channels=False, 
                num_latent_layer=num_latent_layer)
            self.netD = Discriminator_sketch() if "sketch" in self.config["model_type"] else Discriminator_face(input_resolution=face_shape[0])
            self.forward = self.VAE_forward

    def VAE_forward(self, x):
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

    def cycle_save(self, checkpoint_path):
        '''
        'sketch_decoder' refers to the decoder of the face2sketch network
                        and vice versa. 
        '''
        state = {}
        state['sketch_encoder']       = self.netG_A.enc.state_dict()
        state['sketch_decoder']       = self.netG_B.dec.state_dict()
        state['sketch_discriminator'] = self.netD_A.state_dict()
        state['face_encoder']         = self.netG_B.enc.state_dict()
        state['face_decoder']         = self.netG_A.dec.state_dict()
        state['face_dicriminator']    = self.netD_B.state_dict()

        if self.num_latent_layer != 0:
            state['sketch_latent_layer'] = self.netG_A.latent_layer.state_dict()
            state['face_latent_layer']   = self.netG_B.latent_layer.state_dict()
        
        torch.save(state, checkpoint_path)
    
    def cycle_restore(self, checkpoint_path):
        state = torch.load(checkpoint_path)
        self.netG_A.enc.load_state_dict(state['sketch_encoder'])
        self.netG_B.dec.load_state_dict(state['sketch_decoder'])
        self.netD_A.load_state_dict(state['sketch_discriminator'])
        self.netG_B.enc.load_state_dict(state['face_encoder'])
        self.netG_A.dec.load_state_dict(state['face_decoder'])
        self.netD_B.load_state_dict(state['face_dicriminator'])
        
        if self.num_latent_layer != 0:
            self.netG_A.latent_layer.load_state_dict(state['sketch_latent_layer'])
            self.netG_B.latent_layer.load_state_dict(state['face_latent_layer'])
