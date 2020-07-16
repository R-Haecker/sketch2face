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

class Iterator(TemplateIterator):
    def __init__(self, config, root, model, *args, **kwargs):
        super().__init__(config, root, model, *args, **kwargs)
        self.logger = get_logger("Iterator")
        assert config["model_type"] != "sketch2face", "This iterator does not support sketch2face models only single GAN models supported."
        assert self.config["model"] == "model.cycle_gan2.CycleGAN_Model", "This iterator only supports the model: cycle_gan2.CycleGAN_Model"
        # get the config and the logger
        self.config = config
        self.set_random_state()
        self.batch_size = config['batch_size']
        # Check if cuda is available
        self.device = self.set_gpu()
        self.check_cuda(bool(self.device == torch.device("cuda")))
        self.logger.debug(f"Model will pushed to the device: {self.device}")
        # Log the architecture of the model
        self.logger.debug(f"{model}")
        self.G = self.model.netG
        self.D = self.model.NetD
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
        self.critic_iter = 5
        self.lambda_term = 10

        self.cur_iter = 0
    
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
        if self.cuda:
            return Variable(arg).cuda(self.cuda_index)
        else:
            return Variable(arg)

    def check_cuda(self, cuda_flag=False):
        print(cuda_flag)
        if cuda_flag:
            self.cuda_index = 0
            self.cuda = True
            self.D.cuda(self.cuda_index)
            self.G.cuda(self.cuda_index)
            print("Cuda enabled flag: {}".format(self.cuda))
        else:
            self.cuda = False

    def criterion(input_images, )

    def step_op(self, model, **kwargs):
        losses = {}
        losses["discriminator"] = {}
        losses["generator"] = {}
        losses["generator"]["update"] = 0
        losses["generator"]["adv"] = 0
        losses["generator"]["total"] = losses["generator"]["adv"]

        # Requires grad, Generator requires_grad = False
        for p in self.D.parameters():
            p.requires_grad = True

        one = torch.tensor(1, dtype=torch.float)
        mone = one * -1
        if self.cuda:
            one = one.cuda(self.cuda_index)
            mone = mone.cuda(self.cuda_index)

        d_loss_real = 0
        d_loss_fake = 0
        Wasserstein_D = 0
        # Train Dicriminator forward-loss-backward-update self.critic_iter times while 1 Generator forward-loss-backward-update
        self.D.zero_grad()

        input_images = kwargs["image_{}".format(self.config["model_type"])]
        input_images = torch.from_numpy(input_images)
        if (input_images.size()[0] != self.batch_size):
            self.logger.error("Batch size is not as expected")

        #z = torch.rand((self.batch_size, 100, 1, 1))

        input_images = self.get_torch_variable(input_images) 
        
        # Train discriminator
        # WGAN - Training discriminator more iterations than generator
        # Train with real images
        d_loss_real = self.D(input_images).mean()
        
        losses["discriminator"]["real"] = d_loss_real
        # Train with fake images
        z = self.get_torch_variable(torch.randn(self.batch_size, 100, 1, 1))
        

        output_images = self.G(input_images)
        d_loss_fake = self.D(output_images).mean()
        
        losses["discriminator"]["fake"] = d_loss_fake
        
        # Train with gradient penalty
        gradient_penalty = self.calculate_gradient_penalty(input_images.data, output_images.data)
        
        losses = self.criterion()
        

        losses["discriminator"]["gradient_penalty"] = gradient_penalty
        
        d_loss = d_loss_fake - d_loss_real + gradient_penalty
        Wasserstein_D = d_loss_real - d_loss_fake
        losses["discriminator"]["total"] = d_loss
        losses["discriminator"]["Wasserstein_D"] = Wasserstein_D
        losses["discriminator"]["outputs_real"] = losses["discriminator"]["real"]
        losses["discriminator"]["outputs_fake"] = losses["discriminator"]["fake"]
        losses["discriminator"]["update"] = 1
        
        def train_op():
            d_loss_real.backward(mone)
            d_loss_fake.backward(one)

            gradient_penalty.backward()

            self.d_optimizer.step()
            self.logger.info(f'  Discriminator loss_fake: {d_loss_fake}, loss_real: {d_loss_real}')
        
            self.cur_iter = (self.cur_iter +1)%self.critic_iter
            if self.cur_iter == 0:
                # Generator update
                losses["generator"]["update"] = 1
                for p in self.D.parameters():
                    p.requires_grad = False  # to avoid computation

                self.G.zero_grad()
                # train generator
                # compute loss with fake images
                z = self.get_torch_variable(torch.randn(self.batch_size, 100, 1, 1))
                output_images = self.G(z)
                g_loss = self.D(output_images)
                g_loss = g_loss.mean()
                losses["generator"]["adv"] = g_loss
                g_loss.backward(mone)
                g_cost = -g_loss
                self.g_optimizer.step()
                self.logger.info(f'Generator g_loss: {g_loss}')
                losses["generator"]["total"] = losses["generator"]["adv"]

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
        if self.cuda:
            eta = eta.cuda(self.cuda_index)
        else:
            eta = eta

        interpolated = eta * real_images + ((1 - eta) * output_images)

        if self.cuda:
            interpolated = interpolated.cuda(self.cuda_index)
        else:
            interpolated = interpolated

        # define it to calculate gradient
        interpolated = Variable(interpolated, requires_grad=True)

        # calculate probability of interpolated examples
        prob_interpolated = self.D(interpolated)

        # calculate gradients of probabilities with respect to examples
        gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(
                                   prob_interpolated.size()).cuda(self.cuda_index) if self.cuda else torch.ones(
                                   prob_interpolated.size()),
                               create_graph=True, retain_graph=True)[0]

        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_term
        return grad_penalty

    def save(self, checkpoint_path):
        state = {}
        state["generator"] = self.G.state_dict()
        state["discriminator"] = self.D.state_dict()
        state["d_optimizer"] = self.d_optimizer.state_dict()
        state["g_optimizer"] = self.g_optimizer.state_dict()
        torch.save(state, checkpoint_path)
        self.logger.info('Models saved')
    
    def load(self, checkpoint_path):
        state = torch.load(checkpoint_path)
        self.G.load_state_dict(state["generator"])
        self.D.load_state_dict(state["discriminator"])
        self.d_optimizer.load_state_dict(state["d_optimizer"])
        self.g_optimizer.load_state_dict(state["g_optimizer"])
        
    def restore(self, checkpoint_path):
        state = torch.load(checkpoint_path)
        self.G.load_state_dict(state["generator"])
        self.D.load_state_dict(state["discriminator"])
        self.d_optimizer.load_state_dict(state["d_optimizer"])
        self.g_optimizer.load_state_dict(state["g_optimizer"])