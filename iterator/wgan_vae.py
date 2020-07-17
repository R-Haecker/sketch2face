import os
import numpy as np
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import autograd

from edflow import TemplateIterator, get_logger
from edflow.hooks.checkpoint_hooks.common import get_latest_checkpoint
from edflow.util import walk

import itertools
from iterator.util import (
    get_loss_funct,
    np2pt,
    pt2np,
    weights_init
) 

# TODO implemet KLD
# TODO implemet sampling while training

class Iterator(TemplateIterator):
    def __init__(self, config, root, model, *args, **kwargs):
        super().__init__(config, root, model, *args, **kwargs)
        self.logger = get_logger("Iterator")
        assert config["model_type"] != "sketch2face", "This iterator does not support sketch2face models only single GAN models supported."
        assert self.config["model"] == "model.wgan_gradient_penalty.CycleWGAN_GP_VAE", "This iterator only supports the model: wgan_gradient_penalty.CycleWGAN_GP_VAE"
        # get the config and the logger
        self.config = config
        self.set_random_state()
        self.batch_size = config['batch_size']
        # Check if cuda is available
        self.device = self.set_gpu()
        self.logger.debug(f"Model will pushed to the device: {self.device}")
        # Log the architecture of the model
        self.logger.debug(f"{model}")
        self.G = self.model.netG.to(self.device)
        self.D = self.model.netD.to(self.device)
        # WGAN values from paper
        self.b1 = 0.5
        self.b2 = 0.999
        self.learning_rate = config["learning_rate"]
        self.batch_size = self.config["batch_size"]
        # WGAN_gradient penalty uses ADAM
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))

        # Set the logger
        self.number_of_images = 10

        self.generator_iters = self.config["num_steps"]
        self.critic_iter = self.config["losses"]["update_disc"] if "update_disc" in self.config["losses"] else 5
        
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
        
    def D_criterion(self, input_images):
        losses = {}
        losses["discriminator"] = {}
        gp_weight = self.config["losses"]["gp_weight"] if "gp_weight" in self.config["losses"] else 10
        # Requires grad, Generator requires_grad = False
        for p in self.D.parameters():
            p.requires_grad = True
        # Train Dicriminator forward-loss-backward-update self.critic_iter times while 1 Generator forward-loss-backward-update
        # Discriminator #
        '''
        at the moment no sampling
        # Train with fake images
        z = self.get_torch_variable(torch.randn(self.batch_size, 100, 1, 1))
        generated_images = self.G.sample(z)
        '''
        # Train with real images
        losses["discriminator"]["real"] = self.D(input_images).mean()
        #losses["discriminator"]["real"] = d_loss_real
        output_images = self.G(input_images)
        losses["discriminator"]["fake"] = self.D(output_images).mean()
        #d_loss_fake
        
        # Train with gradient penalty
        losses["discriminator"]["gradient_penalty"] = gp_weight * self.calculate_gradient_penalty(input_images.data, output_images.data)
        
        losses["discriminator"]["total"] = losses["discriminator"]["fake"] - losses["discriminator"]["real"] + losses["discriminator"]["gradient_penalty"]
        losses["discriminator"]["Wasserstein_D"] = losses["discriminator"]["real"] - losses["discriminator"]["fake"]
        losses["discriminator"]["outputs_real"] = losses["discriminator"]["real"]
        losses["discriminator"]["outputs_fake"] = losses["discriminator"]["fake"]
        losses["discriminator"]["update"] = 1
        return losses, output_images

    def G_criterion(self, input_images):
        losses = {}
        losses["generator"] = {}
        losses["generator"]["update"] = 0
        losses["generator"]["adv"]    = 0
        losses["generator"]["rec"]    = 0
        losses["generator"]["total"]  = 0
        self.update_G = bool(((self.get_global_step())%self.critic_iter) == (self.critic_iter-1))
        if self.update_G:
            # Generator update
            reconstruction_criterion = get_loss_funct(self.config["losses"]["reconstruction_loss"])
            rec_weight = self.config["losses"]["reconstruction_weight"] if "reconstruction_weight" in self.config["losses"] else 1
            adv_weight = self.config["losses"]["adversarial_weight"] if "adversarial_weight" in self.config["losses"] else 1
            
            losses["generator"]["update"] = 1
            for p in self.D.parameters():
                    p.requires_grad = False  # to avoid computation
            self.G.zero_grad()    
            '''
            maybe sampling later
            # compute loss with fake images
            z = self.get_torch_variable(torch.randn(self.batch_size, 100, 1, 1))
            '''
            output_images = self.G(input_images)
            losses["generator"]["rec"] = rec_weight * torch.mean( reconstruction_criterion( output_images, input_images))
            losses["generator"]["adv"] = adv_weight * self.D(output_images).mean()
            
            losses["generator"]["total"] = losses["generator"]["adv"] + losses["generator"]["rec"]
        return losses

    def step_op(self, model, **kwargs):
        input_images = kwargs["image_{}".format(self.config["model_type"])]
        input_images = torch.from_numpy(input_images)
        if (input_images.size()[0] != self.batch_size):
            self.logger.error("Batch size is not as expected")
        input_images = self.get_torch_variable(input_images) 
        one = torch.tensor(1, dtype=torch.float)
        mone = one * -1
        one = one.to(self.device)
        mone = mone.to(self.device)
    
        self.D.zero_grad()
        d_losses, output_images = self.D_criterion(input_images)
        self.G.zero_grad()
        g_losses = self.G_criterion(input_images)
        losses = dict(d_losses, **g_losses)

        def train_op():
            #self.D.zero_grad()
            losses["discriminator"]["real"].backward(mone)
            losses["discriminator"]["fake"].backward(one)
            losses["discriminator"]["gradient_penalty"].backward()
            self.d_optimizer.step()
            
            if self.update_G:
                '''
                for p in self.D.parameters():
                    p.requires_grad = False  # to avoid computation
                self.G.zero_grad()
                '''
                
                losses["generator"]["rec"].backward(one,retain_graph=True)
                losses["generator"]["adv"].backward(mone)
                self.g_optimizer.step()

        def log_op():
            logs = self.prepare_logs(losses, input_images, output_images)
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
        prob_interpolated = self.D(interpolated)

        # calculate gradients of probabilities with respect to examples
        gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(
                                   prob_interpolated.size()).to(self.device),
                               create_graph=True, retain_graph=True)[0]

        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return grad_penalty

    def save(self, checkpoint_path):
        state = {}
        state["generator"] = self.G.state_dict()
        state["discriminator"] = self.D.state_dict()
        state["d_optimizer"] = self.d_optimizer.state_dict()
        state["g_optimizer"] = self.g_optimizer.state_dict()
        torch.save(state, checkpoint_path)
        self.logger.info('Models saved')
        
    def restore(self, checkpoint_path):
        state = torch.load(checkpoint_path)
        self.G.load_state_dict(state["generator"])
        self.D.load_state_dict(state["discriminator"])
        self.d_optimizer.load_state_dict(state["d_optimizer"])
        self.g_optimizer.load_state_dict(state["g_optimizer"])