import torch
from torch import nn
import numpy as np

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


class WGAN_Discriminator(torch.nn.Module):
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

class Discriminator_WGAN_sketch(WGAN_Discriminator):
    def __init__(self):
        super().__init__(input_channels = 1, input_resolution = 32)

class Discriminator_WGAN_face(WGAN_Discriminator):
    def __init__(self, input_resolution):
        super().__init__(input_channels = 3, input_resolution = input_resolution)
