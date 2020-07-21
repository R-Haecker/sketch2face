import os
import numpy as np
import random
import itertools

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import autograd

from edflow import TemplateIterator, get_logger
from edflow.hooks.checkpoint_hooks.common import get_latest_checkpoint
from edflow.util import walk

from model.vae2 import VAE_Model
from iterator.util import (
    get_loss_funct,
    np2pt,
    pt2np,
    weights_init
) 

############################
###  Cycle GAN Iterator  ###
############################

class Iterator_CycleGAN(TemplateIterator):
    def __init__(self, config, root, model, *args, **kwargs):
        super().__init__(config, root, model, *args, **kwargs)
        self.logger = get_logger("Iterator")
        assert self.config["model"] == "model.wgan.CycleWGAN_GP_VAE", "This iterator only supports the model: wgan_gradient_penalty.CycleWGAN_GP_VAE"
        # get the config and the logger
        self.config = config
        self.set_random_state()
        self.batch_size = config['batch_size']
        # Check if cuda is available
        self.device = self.set_gpu()
        self.logger.debug(f"Model will pushed to the device: {self.device}")
        # Log the architecture of the model
        self.logger.debug(f"{model}")
        self.model  = model.to(self.device)
        # WGAN values from paper
        self.b1 = 0.5
        self.b2 = 0.999
        self.learning_rate = config["learning_rate"]
        self.batch_size = self.config["batch_size"]
        self.critic_iter = self.config["losses"]["update_disc"] if "update_disc" in self.config["losses"] else 5
        
        self.load_pretrained_vaes()
        
        self.add_latent_layer = bool('num_latent_layer' in self.config['variational'] and self.config['variational']['num_latent_layer'] > 0)
        self.only_latent_layer = bool('only_latent_layer' in self.config['optimization'] and self.config['optimization']['only_latent_layer'])
        if self.only_latent_layer:
            self.critic_iter = 1
            self.logger.debug("critic_iter set to 1 since only_latent_layer is True.")
            self.optimizer_Lin = torch.optim.Adam(itertools.chain(self.model.netG_A.latent_layer.parameters(), self.model.netG_B.latent_layer.parameters()), lr=self.config["learning_rate"])
            self.logger.debug("Only latent layers are optimized\nNumber of latent layers: {}".format(self.config['variational']['num_latent_layer']))
        else:
            # WGAN_gradient penalty uses ADAM
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.model.netG_A.parameters(), self.model.netG_B.parameters()), lr=self.learning_rate, betas=(self.b1, self.b2))
        D_lr_factor = self.config["optimization"]["D_lr_factor"] if "D_lr_factor" in config["optimization"] else 1
        self.optimizer_D = torch.optim.Adam(itertools.chain(self.model.netD_A.parameters(), self.model.netD_B.parameters()), lr=D_lr_factor * self.config["learning_rate"], betas=(self.b1, self.b2))

    def set_gpu(self):
        """Move the model to device cuda if available and use the specified GPU"""
        if "CUDA_VISIBLE_DEVICES" in self.config:
            if type(self.config["CUDA_VISIBLE_DEVICES"]) != str:
                self.config["CUDA_VISIBLE_DEVICES"] = str(self.config["CUDA_VISIBLE_DEVICES"])
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = self.config["CUDA_VISIBLE_DEVICES"]
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_pretrained_vaes(self):
        log_string = "No models loaded"
        if "load_models" in self.config:
            if "sketch_path" in self.config["load_models"] and "face_path" in self.config["load_models"]:
                #load state dict of components of the VAE's
                sketch_state = torch.load(self.config["load_models"]["sketch_path"])
                face_state = torch.load(self.config["load_models"]["face_path"])
                self.model.netG_A.enc.load_state_dict(sketch_state['encoder'])
                self.model.netG_A.dec.load_state_dict(face_state['decoder'])
                self.model.netD_A.load_state_dict(sketch_state['discriminator'])
                self.model.netG_B.enc.load_state_dict(face_state['encoder'])
                self.model.netG_B.dec.load_state_dict(sketch_state['decoder'])
                self.model.netD_B.load_state_dict(face_state['discriminator'])
                log_string = "Sketch VAE loaded from {}\nFace VAE loaded from {}".format(self.config["load_models"]["sketch_path"], self.config["load_models"]["face_path"])
        self.logger.debug(log_string)

    def set_random_state(self):
        np.random.seed(self.config["random_seed"])
        torch.random.manual_seed(self.config["random_seed"])

    def get_torch_variable(self, arg):
        return Variable(arg).to(self.device)
        
    def criterion(self):
        """This function returns a dictionary with all neccesary losses for the model."""
        if "sketch" in self.config["losses"]["reconstruction_loss"]:
            sketch_rec_crit = get_loss_funct(self.config["losses"]["reconstruction_loss"]["sketch"])
        if "face" in self.config["losses"]["reconstruction_loss"]:
            face_rec_crit = get_loss_funct(self.config["losses"]["reconstruction_loss"]["face"])
        else:
            face_rec_crit = sketch_rec_crit = get_loss_funct(self.config["losses"]["reconstruction_loss"])
        
        rec_weight = self.config["losses"]["reconstruction_weight"]
        adv_weight = self.config["losses"]["adversarial_weight"]
        gp_weight = self.config["losses"]["gp_weight"] if "gp_weight" in self.config["losses"] else 0

        losses = {}

        kld_weight = self.config["losses"]["kld"]["weight"] if "kld" in self.config["losses"] else 0
        kld_delay = self.config["losses"]['kld']["delay"] if "delay" in self.config["losses"]['kld'] else 0
        kld_slope_steps = self.config["losses"]['kld']["slope_steps"] if "slope_steps" in self.config["losses"]['kld'] else 0
        losses["kld_weight"] = self.kld_update(kld_weight, self.get_global_step(), kld_delay, kld_slope_steps)

        #########################
        ###  A: cycle sketch  ###
        #########################
        # Generator
        self.update_G = bool(((self.get_global_step())%self.critic_iter) == 0)
        if self.update_G:
            losses["sketch_cycle"] = {}
            losses["update_generator"] = 1
            losses["sketch_cycle"]["rec"] = rec_weight * torch.mean( sketch_rec_crit( self.model.output['real_A'], self.model.output['rec_A'] ) )
            losses["sketch_cycle"]["adv"] = adv_weight * torch.mean( self.model.netD_B(self.model.output['fake_B']).view(-1) )
            losses["sketch_cycle"]["generator_total"] = losses["sketch_cycle"]["rec"] - losses["sketch_cycle"]["adv"]
            if kld_weight > 0 and "sigma" in self.config["variational"] and self.config["variational"]["sigma"]:
                losses["sketch_cycle"]["kld"] = losses["kld_weight"]* -0.5 * torch.mean(1 + self.model.netG_A.logvar - self.model.netG_A.mu.pow(2) - self.model.netG_A.logvar.exp())
                losses["sketch_cycle"]["generator_total"] += losses["sketch_cycle"]["kld"]
        else:
            losses["update_generator"] = 0
            losses["sketch_cycle"] = self.losses_sketch_cycle
        # Discriminator
        losses["sketch_cycle"]["disc_fake"] = self.model.netD_B( self.model.output['fake_B'].detach()).mean()
        losses["sketch_cycle"]["disc_real"] = self.model.netD_B( self.model.output['real_B'].detach()).mean()
        losses["discriminator_face"] = {}
        losses["discriminator_face"]["outputs_fake"] = losses["sketch_cycle"]["disc_fake"].clone().detach().cpu().numpy()
        losses["discriminator_face"]["outputs_real"] = losses["sketch_cycle"]["disc_real"].clone().detach().cpu().numpy()
        
        losses["sketch_cycle"]["disc_total"] = losses["sketch_cycle"]["disc_fake"] - losses["sketch_cycle"]["disc_real"]
        if gp_weight > 0:
            losses["sketch_cycle"]["gp"] = gp_weight * self.calculate_gradient_penalty(self.model.netD_B, self.model.output['real_B'], self.model.output['fake_B'].detach())
            losses["sketch_cycle"]["disc_total"] += losses["sketch_cycle"]["gp"]

        self.logger.debug('losses["sketch_cycle"]["disc_real"] ' + str(losses["sketch_cycle"]["disc_real"]))
        self.logger.debug('losses["sketch_cycle"]["disc_total"] ' + str(losses["sketch_cycle"]["disc_total"]))
        
        #######################
        ###  B: cycle face  ###
        #######################
        # Generator
        if self.update_G:    
            losses["face_cycle"] = {}
            losses["face_cycle"]["rec"] = rec_weight * torch.mean( face_rec_crit( self.model.output['real_B'], self.model.output['rec_B'] ) )
            losses["face_cycle"]["adv"] = adv_weight * torch.mean( self.model.netD_A(self.model.output['fake_A']).view(-1) )
            losses["face_cycle"]["generator_total"] = losses["face_cycle"]["rec"] - losses["face_cycle"]["adv"]
            if kld_weight > 0 and self.model.sigma:
                losses["face_cycle"]["kld"] = losses["kld_weight"]* -0.5 * torch.mean(1 + self.model.netG_B.logvar - self.model.netG_B.mu.pow(2) - self.model.netG_B.logvar.exp())
                losses["face_cycle"]["generator_total"] += losses["face_cycle"]["kld"]
        else:
            losses["face_cycle"] = self.losses_face_cycle
        # Discriminator
        losses["face_cycle"]["disc_fake"] = self.model.netD_A( self.model.output['fake_A'].detach()).mean()
        losses["face_cycle"]["disc_real"] = self.model.netD_A( self.model.output['real_A'].detach()).mean()
        losses["discriminator_sketch"] = {}
        losses["discriminator_sketch"]["outputs_fake"] = losses["face_cycle"]["disc_fake"].clone().detach().cpu().numpy()
        losses["discriminator_sketch"]["outputs_real"] = losses["face_cycle"]["disc_real"].clone().detach().cpu().numpy()
        losses["face_cycle"]["disc_total"] = losses["face_cycle"]["disc_fake"] - losses["face_cycle"]["disc_real"]
        if gp_weight > 0:
            losses["face_cycle"]["gp"] = gp_weight * self.calculate_gradient_penalty(self.model.netD_A, self.model.output['real_A'], self.model.output['fake_A'].detach())
            losses["face_cycle"]["disc_total"] += losses["face_cycle"]["gp"]
        
        # Generator losses
        if self.update_G:
            losses["generators"]  = losses["face_cycle"]["generator_total"] + losses["sketch_cycle"]["generator_total"]
            self.losses_generator = losses["generators"]
            self.losses_sketch_cycle = {x: losses["sketch_cycle"][x] for x in losses["sketch_cycle"] if x not in ["disc_fake", "disc_real", "disc_total", "gp"]}
            self.losses_face_cycle   = {x: losses["face_cycle"][x]   for x in losses["face_cycle"]   if x not in ["disc_fake", "disc_real", "disc_total", "gp"]}
        else:
            losses["generators"] = self.losses_generator
            
        # Discriminator losses
        losses["discriminators"] = losses["sketch_cycle"]["disc_total"] + losses["face_cycle"]["disc_total"]
        
        return losses

    def step_op(self, model, **kwargs):
        # get inputs
        real_A = kwargs["image_sketch"]
        real_B = kwargs["image_face"]
    
        real_A = self.get_torch_variable(torch.tensor(real_A))
        real_B = self.get_torch_variable(torch.tensor(real_B))

        self.logger.debug("sketch_images.shape: " + str(real_A.shape))
        self.logger.debug("face_images.shape: " + str(real_B.shape))
        output_images = self.model(real_A, real_B)
        self.logger.debug("fake_face.shape: " + str(output_images[0].shape))
        self.logger.debug("fake_sketch.shape: " + str(output_images[1].shape))
        
        # create all losses
        losses = self.criterion()

        def train_op():
            # This function will be executed if the model is in training mode
            if "optimization" in self.config and "reduce_lr" in self.config["optimization"]:
                # reduce the learning rate if specified
                losses["current_learning_rate"], losses["amplitude_learning_rate"] = self.update_learning_rate()
            # update the generators
            if self.update_G and not self.only_latent_layer:
                self.set_requires_grad([self.model.netD_A, self.model.netD_B], False)
                self.optimizer_G.zero_grad()
                losses["generators"].backward()
                self.optimizer_G.step()
                print("GHJKJHGFG")
            # train only linear layers
            if self.only_latent_layer:
                self.set_requires_grad([self.model.netG_A.enc, self.model.netG_A.dec, self.model.netG_B.enc, self.model.netG_B.dec], False)
                self.optimizer_Lin.zero_grad()
                losses["generators"].backward()
                self.optimizer_Lin.step()
                #self.train_lin_layer(losses)
            # update the discriminators
            self.set_requires_grad([self.model.netD_A, self.model.netD_B], True)
            self.optimizer_D.zero_grad()
            losses["discriminators"].backward()
            self.optimizer_D.step()
            
        def log_op():
            logs = self.prepare_logs(losses, [real_A, real_B], output_images)
            return logs

        def eval_op():
            return {}

        return {"train_op": train_op, "log_op": log_op, "eval_op": eval_op}
    
    def train_lin_layer(self, losses):
        self.set_requires_grad([self.model.netG_A.enc, self.model.netG_A.dec, self.model.netG_B.enc, self.model.netG_B.dec], False)
        self.optimizer_Lin.zero_grad()
        losses["generators"].backward()
        self.optimizer_Lin.step()

    def prepare_logs(self, losses, inputs, predictions):
        """Return a log dictionary with all instersting data to log."""
        # create a dictionary to log with all interesting variables 
        logs = {
            "images": {},
            "scalars":{
                **losses
                }
        }
        ############
        ## images ##
        ############
        real_A_img = pt2np(inputs[0])
        real_B_img = pt2np(inputs[1])
        
        logs["images"].update({"batch_input_sketch": real_A_img})
        logs["images"].update({"batch_input_face": real_B_img})

        fake_A_img = pt2np(predictions[1])
        fake_B_img = pt2np(predictions[0])

        logs["images"].update({"batch_fake_sketch": fake_A_img})
        logs["images"].update({"batch_fake_face": fake_B_img})
        
        rec_A_img  = pt2np(self.model.output['rec_A'])
        rec_B_img  = pt2np(self.model.output['rec_B'])
        
        logs["images"].update({"batch_rec_sketch": rec_A_img})
        logs["images"].update({"batch_rec_face": rec_B_img})
        
        # log only max three images separately
        max_num = 3 if self.config["batch_size"] > 3 else self.config["batch_size"]
        for i in range(max_num):
            logs["images"].update({"input_sketch_" + str(i): np.expand_dims(real_A_img[i],0)})
            logs["images"].update({"input_face_"   + str(i): np.expand_dims(real_B_img[i],0)})
            logs["images"].update({"fake_sketch_"  + str(i): np.expand_dims(fake_A_img[i],0)})
            logs["images"].update({"fake_face_"    + str(i): np.expand_dims(fake_B_img[i],0)})
            
        def conditional_convert2np(log_item):
            if isinstance(log_item, torch.Tensor):
                log_item = log_item.detach().cpu().numpy()
            return log_item
        # convert to numpy
        walk(logs, conditional_convert2np, inplace=True)
        return logs
    
    def set_requires_grad(self, nets, requires_grad=False): # this function was copied from: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/base_model.py 
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def calculate_gradient_penalty(self, discriminator, real_images, output_images):
        eta = torch.FloatTensor(self.batch_size,1,1,1).uniform_(0,1)
        eta = eta.expand(self.batch_size, real_images.size(1), real_images.size(2), real_images.size(3))
        self.logger.debug("real_images.shape: " + str(real_images.shape))
        self.logger.debug("fake_images.shape: " + str(output_images.shape))
        self.logger.debug("eta.shape: " + str(eta.shape))
        eta = eta.to(self.device)
        
        interpolated = eta * real_images + ((1 - eta) * output_images)

        interpolated = interpolated.to(self.device)

        # define it to calculate gradient
        interpolated = Variable(interpolated, requires_grad=True)

        # calculate probability of interpolated examples
        prob_interpolated = discriminator(interpolated)

        # calculate gradients of probabilities with respect to examples
        gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(
                                   prob_interpolated.size()).to(self.device),
                               create_graph=True, retain_graph=True)[0]

        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return grad_penalty

    def update_learning_rate(self):
        step = torch.tensor(self.get_global_step(), dtype = torch.float)
        num_step = self.config["num_steps"]
        current_ratio = step/self.config["num_steps"]
        reduce_lr_ratio = self.config["optimization"]["reduce_lr"]
        if current_ratio >= self.config["optimization"]["reduce_lr"]:
            def amplitide_lr(step):
                delta = (1-reduce_lr_ratio)*num_step
                return (num_step-step)/delta
            amp = amplitide_lr(step)
            lr = self.config["learning_rate"] * amp
            
            # Update the learning rates
            for g in self.optimizer_G.param_groups:
                g['lr'] = lr
            
            D_lr_factor = self.config["optimization"]["D_lr_factor"] if "D_lr_factor" in self.config["optimization"] else 1
            for g in self.optimizer_D.param_groups:
                g['lr'] = lr*D_lr_factor
            return lr, amp
        else:
            return self.config["learning_rate"], 1

    def kld_update(self, weight, steps, delay, slope_steps):
        output = 0
        if steps >= delay and steps < slope_steps+delay:
            output = weight*(1 - np.cos((steps-delay)*np.pi/slope_steps))/2
        if steps >= slope_steps+delay:
            output = weight
        return output

    def save(self, checkpoint_path):
        '''
        'sketch_decoder' refers to the decoder of the face2sketch network
                        and vice versa. 
        '''
        state = {}
        state['sketch_encoder'] = self.model.netG_A.enc.state_dict()
        state['sketch_decoder'] = self.model.netG_B.dec.state_dict()
        state['sketch_discriminator'] = self.model.netD_A.state_dict()
        state['face_encoder'] = self.model.netG_B.enc.state_dict()
        state['face_decoder'] = self.model.netG_A.dec.state_dict()
        state['face_dicriminator'] = self.model.netD_B.state_dict()
        state['optimizer_D'] = self.optimizer_D.state_dict()
        if self.add_latent_layer:
            state['sketch_latent_layer'] = self.model.netG_A.latent_layer.state_dict()
            state['face_latent_layer'] = self.model.netG_B.latent_layer.state_dict()
        if self.only_latent_layer:
            state['optimizer_Lin'] = self.optimizer_Lin.state_dict()
        else:
            state['optimizer_G'] = self.optimizer_G.state_dict()

        
        torch.save(state, checkpoint_path)
    
    def restore(self, checkpoint_path):
        state = torch.load(checkpoint_path)
        self.model.netG_A.enc.load_state_dict(state['sketch_encoder'])
        self.model.netG_B.dec.load_state_dict(state['sketch_decoder'])
        self.model.netD_A.load_state_dict(state['sketch_discriminator'])
        self.model.netG_B.enc.load_state_dict(state['face_encoder'])
        self.model.netG_A.dec.load_state_dict(state['face_decoder'])
        self.model.netD_B.load_state_dict(state['face_discriminator'])
        self.optimizer_D.load_state_dict(state['optimizer_D'])

        if self.add_latent_layer:
            self.model.netG_A.latent_layer.load_state_dict(state['sketch_latent_layer'])
            self.model.netG_B.latent_layer.load_state_dict(state['face_latent_layer'])
        if self.only_latent_layer:
            self.optimizer_Lin.load_state_dict(state['optimizer_Lin'])
        else:
            self.optimizer_G.load_state_dict(state['optimizer_G'])


####################
### VAE Iterator ###
####################

class Iterator_VAE(TemplateIterator):
    def __init__(self, config, root, model, *args, **kwargs):
        super().__init__(config, root, model, *args, **kwargs)
        self.logger = get_logger("Iterator")
        assert config["model_type"] != "sketch2face", "This iterator does not support sketch2face models only single GAN models supported."
        assert self.config["model"] == "model.wgan.CycleWGAN_GP_VAE", "This iterator only supports the model: wgan.CycleWGAN_GP_VAE"
        # get the config and the logger
        self.config = config
        self.set_random_state()
        self.batch_size = config['batch_size']
        # Check if cuda is available
        self.device = self.set_gpu()
        self.logger.debug(f"Model will pushed to the device: {self.device}")
        # Log the architecture of the model
        self.logger.debug(f"{model}")
        self.netG = self.model.netG.to(self.device)
        self.netD = self.model.netD.to(self.device)
        # WGAN values from paper
        self.b1 = 0.5
        self.b2 = 0.999
        self.learning_rate = config["learning_rate"]
        self.batch_size = self.config["batch_size"]
        self.critic_iter = self.config["losses"]["update_disc"] if "update_disc" in self.config["losses"] else 5
        # WGAN_gradient penalty uses ADAM
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))
        D_lr_factor = self.config["optimization"]["D_lr_factor"] if "D_lr_factor" in config["optimization"] else 1
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=self.learning_rate*D_lr_factor, betas=(self.b1, self.b2))
        
    def set_gpu(self):
        """Move the model to device cuda if available and use the specified GPU"""
        if "CUDA_VISIBLE_DEVICES" in self.config:
            if type(self.config["CUDA_VISIBLE_DEVICES"]) != str:
                self.config["CUDA_VISIBLE_DEVICES"] = str(self.config["CUDA_VISIBLE_DEVICES"])
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = self.config["CUDA_VISIBLE_DEVICES"]
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def set_random_state(self):
        np.random.seed(self.config["random_seed"])
        torch.random.manual_seed(self.config["random_seed"])

    def get_torch_variable(self, arg):
        return Variable(arg).to(self.device)
    
    def generate_from_sample(self):
        self.logger.debug("Using samples for training")
        z = torch.randn(self.batch_size, self.config["latent_dim"]).to(self.device)
        self.generated_images = self.netG.dec(z)
        return self.generated_images
        
    def D_criterion(self, input_images):
        losses = {}
        losses["discriminator"] = {}
        gp_weight = self.config["losses"]["gp_weight"] if "gp_weight" in self.config["losses"] else 10
        # Requires grad, Generator requires_grad = False
        for p in self.netD.parameters():
            p.requires_grad = True
        # Train Dicriminator forward-loss-backward-update self.critic_iter times while 1 Generator forward-loss-backward-update
        # Discriminator #
        
        
        # Train with real images
        losses["discriminator"]["real"] = self.netD(input_images).mean()
        #losses["discriminator"]["real"] = d_loss_real
        output_images = self.netG(input_images)
        losses["discriminator"]["fake"] = self.netD(output_images.detach()).mean()
        #d_loss_fake
        
        # Train with gradient penalty
        losses["discriminator"]["gradient_penalty"] = gp_weight * self.calculate_gradient_penalty(input_images.data, output_images.data)
        
        losses["discriminator"]["total"] = losses["discriminator"]["fake"] - losses["discriminator"]["real"] + losses["discriminator"]["gradient_penalty"]
        losses["discriminator"]["Wasserstein_D"] = losses["discriminator"]["real"] - losses["discriminator"]["fake"]
        losses["discriminator"]["outputs_real"] = losses["discriminator"]["real"]
        losses["discriminator"]["outputs_fake"] = losses["discriminator"]["fake"]

        if "sample" in self.config["losses"] and self.config["losses"]["sample"]:
            self.logger.debug("Using samples for training")
            z = torch.randn(self.batch_size, self.config["latent_dim"]).to(self.device)
            self.generated_images = self.netG.dec(z)
            losses["discriminator"]["sample"] = self.netD(self.generated_images.detach()).mean()
            losses["discriminator"]["total"] += losses["discriminator"]["sample"]
            losses["discriminator"]["outputs_sample"] = losses["discriminator"]["sample"]
        
        losses["discriminator"]["update"] = 1
        return losses, output_images

    def G_criterion(self, input_images):
        losses = {}
        losses["generator"] = {}
        self.update_G = bool(((self.get_global_step())%self.critic_iter) == 0)
        if self.update_G:            
            kld_weight = self.config["losses"]["kld"]["weight"] if "kld" in self.config["losses"] else 0
            kld_delay = self.config["losses"]['kld']["delay"] if "delay" in self.config["losses"]['kld'] else 0
            kld_slope_steps = self.config["losses"]['kld']["slope_steps"] if "slope_steps" in self.config["losses"]['kld'] else 0
            losses["generator"]["kld_weight"] = self.kld_update(kld_weight, self.get_global_step(), kld_delay, kld_slope_steps)

            if kld_weight > 0 and  losses["generator"]["kld_weight"] > 0 and "sigma" in self.config["variational"] and self.config["variational"]["sigma"]:
                losses["generator"]["kld"] = losses["generator"]["kld_weight"]* -0.5 * torch.mean(1 + self.netG.logvar - self.model.netG.mu.pow(2) - self.netG.logvar.exp())
            else:
                losses["generator"]["kld"] = 0
            # Generator update
            reconstruction_criterion = get_loss_funct(self.config["losses"]["reconstruction_loss"])
            rec_weight = self.config["losses"]["reconstruction_weight"] if "reconstruction_weight" in self.config["losses"] else 1
            adv_weight = self.config["losses"]["adversarial_weight"] if "adversarial_weight" in self.config["losses"] else 1
            
            losses["generator"]["update"] = 1
            for p in self.netD.parameters():
                p.requires_grad = False  # to avoid computation
            self.netG.zero_grad()    
            '''
            maybe sampling later
            # compute loss with fake images
            z = self.get_torch_variable(torch.randn(self.batch_size, 100, 1, 1))
            '''
            output_images = self.netG(input_images)
            losses["generator"]["rec"] = rec_weight * torch.mean( reconstruction_criterion( output_images, input_images))
            losses["generator"]["adv"] = adv_weight * self.netD(output_images).mean()
            
            losses["generator"]["total"] = losses["generator"]["rec"] - losses["generator"]["adv"] + losses["generator"]["kld"]

            if "sample" in self.config["losses"] and self.config["losses"]["sample"]:  
                losses["generator"]["sample"] = self.netD(self.generated_images).mean()
                losses["generator"]["total"] -= adv_weight * losses["generator"]["sample"]  

            self.losses_generator = losses["generator"]

        else:
            losses["generator"] = self.losses_generator
            losses["generator"]["update"] = 0
        return losses

    def step_op(self, model, **kwargs):
        input_images = kwargs["image_{}".format(self.config["model_type"])]
        input_images = torch.from_numpy(input_images)
        if (input_images.size()[0] != self.batch_size):
            self.logger.error("Batch size is not as expected")
        input_images_D = self.get_torch_variable(input_images) 
        input_images_G = self.get_torch_variable(input_images) 
        one = torch.tensor(1, dtype=torch.float)
        mone = one * -1
        one = one.to(self.device)
        mone = mone.to(self.device)
    
        self.netD.zero_grad()
        losses, output_images = self.D_criterion(input_images_D)

        def train_op():
            # This function will be executed if the model is in training mode
            if "optimization" in self.config and "reduce_lr" in self.config["optimization"]:
                # reduce the learning rate if specified
                losses["current_learning_rate"], losses["amplitude_learning_rate"] = self.update_learning_rate()
            # Update the discriminator
            losses["discriminator"]["total"].backward()
            self.optimizer_D.step()
            # Update the generator
            self.netG.zero_grad()
            g_losses = self.G_criterion(input_images_G)
            losses["generator"] = g_losses["generator"]
            if self.update_G:
                losses["generator"]["total"].backward()
                #losses["generator"]["adv"].backward(mone)
                self.optimizer_G.step()

        def log_op():
            logs = self.prepare_logs(losses, input_images_D.detach(), output_images)
            return logs

        def eval_op():
            return {}

        return {"train_op": train_op, "log_op": log_op, "eval_op": eval_op}
    
    def prepare_logs(self, losses, input_images, output_images):
        """Return a log dictionary with all instersting data to log."""
        # create a dictionary to log with all interesting variables 
        logs = {
            "images": {},
            "scalars":{
                **losses
                }
        }
        # images
        input_img = pt2np(input_images)
        logs["images"].update({"batch_input": input_img})
        output_img = pt2np(output_images)
        logs["images"].update({"batch_output": output_img})
        
        # log only max three images separately
        max_num = 3 if self.config["batch_size"] > 3 else self.config["batch_size"]
        for i in range(max_num):
            logs["images"].update({"input_" + str(i): np.expand_dims(input_img[i],0)})
            logs["images"].update({"output_" + str(i): np.expand_dims(output_img[i],0)})
            
        def conditional_convert2np(log_item):
            if isinstance(log_item, torch.Tensor):
                log_item = log_item.detach().cpu().numpy()
            return log_item
        # convert to numpy
        walk(logs, conditional_convert2np, inplace=True)
        return logs
        
    def calculate_gradient_penalty(self, real_images, output_images):
        eta = torch.FloatTensor(self.batch_size,1,1,1).uniform_(0,1)
        eta = eta.expand(self.batch_size, real_images.size(1), real_images.size(2), real_images.size(3))
        self.logger.debug("real_images.shape: " + str(real_images.shape))
        self.logger.debug("fake_images.shape: " + str(output_images.shape))
        self.logger.debug("eta.shape: " + str(eta.shape))
        eta = eta.to(self.device)
        
        interpolated = eta * real_images + ((1 - eta) * output_images)

        interpolated = interpolated.to(self.device)

        # define it to calculate gradient
        interpolated = Variable(interpolated, requires_grad=True)

        # calculate probability of interpolated examples
        prob_interpolated = self.netD(interpolated)

        # calculate gradients of probabilities with respect to examples
        gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(
                                   prob_interpolated.size()).to(self.device),
                               create_graph=True, retain_graph=True)[0]

        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return grad_penalty

    def update_learning_rate(self):
        step = torch.tensor(self.get_global_step(), dtype = torch.float)
        num_step = self.config["num_steps"]
        current_ratio = step/self.config["num_steps"]
        reduce_lr_ratio = self.config["optimization"]["reduce_lr"]
        if current_ratio >= self.config["optimization"]["reduce_lr"]:
            def amplitide_lr(step):
                delta = (1-reduce_lr_ratio)*num_step
                return (num_step-step)/delta
            amp = amplitide_lr(step)
            lr = self.config["learning_rate"] * amp
            
            # Update the learning rates
            for g in self.optimizer_G.param_groups:
                g['lr'] = lr
            
            D_lr_factor = self.config["optimization"]["D_lr_factor"] if "D_lr_factor" in self.config["optimization"] else 1
            for g in self.optimizer_D.param_groups:
                g['lr'] = lr*D_lr_factor
            return lr, amp
        else:
            return self.config["learning_rate"], 1

    def kld_update(self, weight, steps, delay, slope_steps):
        output = 0
        if steps >= delay and steps < slope_steps+delay:
            output = weight*(1 - np.cos((steps-delay)*np.pi/slope_steps))/2
        if steps > slope_steps+delay:
            output = weight
        return output


    def save(self, checkpoint_path):
        state = {}
        state["encoder"] = self.netG.enc.state_dict()
        state["decoder"] = self.netG.dec.state_dict()
        state["discriminator"] = self.netD.state_dict()
        state["optimizer_D"] = self.optimizer_D.state_dict()
        state["optimizer_G"] = self.optimizer_G.state_dict()
        torch.save(state, checkpoint_path)
        self.logger.info('Models saved')
        
    def restore(self, checkpoint_path):
        state = torch.load(checkpoint_path)
        self.netG.enc.load_state_dict(state["encoder"])
        self.netG.dec.load_state_dict(state["decoder"])
        self.netD.load_state_dict(state["discriminator"])
        self.optimizer_D.load_state_dict(state["optimizer_D"])
        self.optimizer_G.load_state_dict(state["optimizer_G"])