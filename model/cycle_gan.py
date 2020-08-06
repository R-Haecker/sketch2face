import torch
from torch import nn
from model.vae import VAE
from model.discriminator import Discriminator_sketch, Discriminator_face

from edflow.custom_logging import LogSingleton
from edflow import get_logger

from model.util import set_random_state

###################
###  Cycle GAN  ###
###################

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

####################
###  Cycle WGAN  ###
####################

class Cycle_WGAN(nn.Module):
    def __init__(self, config):
        super(Cycle_WGAN, self).__init__()
        
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
            self.netG = VAE(
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