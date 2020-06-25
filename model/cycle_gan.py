import torch
from torch import nn
from model.vae import VAE_Model
from model.discriminator import Discriminator_sketch, Discriminator_face

class CycleGAN_Model(nn.Module):
    def __init__(self, config):
        self.config = config
        self.output_names = ['real_A', 'fake_B', 'rec_A', 'real_B', 'fake_A', 'rec_B']
        self.cycle = "sketch" in self.config["model_type"] and "face" in self.config["model_type"]
        if self.cycle:
            self.netG_A = VAE_Model(self.config, sketch=True)
            self.netD_A = Discriminator_sketch()
            self.netG_B = VAE_Model(self.config, sketch=False)
            self.netD_B = Discriminator_face()
        else:
            sketch = True if "sketch" in self.config["model_type"] else False
            self.netG = VAE_Model(self.config, sketch=sketch)
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
            fake_B = self.netG_A(real_A)
            rec_A = self.netG_B(fake_B)
            
            output_names += self.output_names[:3]
            output_values +=  [real_A, fake_B, rec_A]
            out += [fake_B]
        #forward pass of images of domain B (faces)
        if real_B is not None:    
            fake_A = self.netG_B(real_B) 
            rec_B = self.netG_A(fake_A)

            output_names += output_names[3:]
            output_values += [real_B, fake_A, rec_B]
            out += [fake_A]
        
        self.output = dict(zip(self.output_names, output_values))
        return out

    def forward(self, real_A, real_B=None):
        if self.cycle:
            return self.cycle_forward(real_A, real_B)
        else:
            return self.vae_forward(real_A)