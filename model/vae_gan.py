from torch import nn
from model.vae import VAE_config

from edflow import get_logger

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

class Discriminator_sketch(nn.Module):
    def __init__(self, nc=1, ndf=32, droprate=0, wasserstein=False):
        super(Discriminator_sketch, self).__init__()

        self.main = nn.Sequential(
            # input is (ndf) x 32 x 32
            nn.Conv2d(nc, ndf * 2, 4, 2, 1, bias=False),
            nn.Dropout(droprate),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.Dropout(droprate),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.Dropout(droprate),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid() if not wasserstein else nn.Identity()
        )

    def forward(self, input):
        return self.main(input)

class Discriminator_face(nn.Module):
    def __init__(self, nc=3, ndf=64, droprate=0, wasserstein=False):
        super(Discriminator_face, self).__init__()

        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.Dropout(droprate),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.Dropout(droprate),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.Dropout(droprate),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.Dropout(droprate),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid() if not wasserstein else nn.Identity()
        )

    def forward(self, input):
        return self.main(input)


    

        '''
        self.sigma = self.config['variational']['sigma']
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
        '''