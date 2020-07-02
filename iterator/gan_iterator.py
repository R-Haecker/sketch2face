import os
import numpy as np
import random

import torch
import torch.nn as nn
from torch.autograd import Variable

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
        # export to the right gpu if specified in the config
        self.device = self.set_gpu()
        self.logger.debug(f"Model will pushed to the device: {self.device}")
        # get the config and the logger
        self.config = config
        self.set_random_state()
        self.batch_size = config['batch_size']
        # Config will be tested inside the Model class even for the iterator
        # Log the architecture of the model
        self.logger.debug(f"{model}")
        self.model = model.to(self.device)
        
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.model.netG.parameters()), lr=self.config["learning_rate"]) # betas=(opt.beta1, 0.999))
        D_lr_factor = self.config["optimization"]["D_lr_factor"] if "D_lr_factor" in config["optimization"] else 1
        self.optimizer_D = torch.optim.Adam(self.model.netD.parameters(), lr= D_lr_factor * self.config["learning_rate"]) # betas=(opt.beta1, 0.999))
        
        self.real_labels = torch.ones(self.batch_size, device=self.device)
        self.fake_labels = torch.zeros(self.batch_size, device=self.device)
    
    def set_random_state(self):
        np.random.seed(self.config["random_seed"])
        torch.random.manual_seed(self.config["random_seed"])

    def criterion(self, model_input, model_output):
        """This function returns a dictionary with all neccesary losses for the model."""
        
        reconstruction_criterion = get_loss_funct(self.config["losses"]["reconstruction_loss"])
        rec_weight = self.config["losses"]["reconstruction_weight"]
        adversarial_criterion = get_loss_funct(self.config["losses"]["adversarial_loss"])
        adv_weight = self.config["losses"]["adversarial_weight"]
        losses = {}

        losses["generator"] = {}
        losses["generator"]["rec"] = rec_weight * torch.mean( reconstruction_criterion( model_output, model_input))
        losses["generator"]["adv"] = adv_weight * torch.mean( adversarial_criterion( self.model.netD(model_output).view(-1), self.real_labels))
        losses["generator"]["total"] = losses["generator"]["rec"] + losses["generator"]["adv"]

        netD_real_outputs = self.model.netD( model_input.detach()).view(-1)
        netD_fake_outputs = self.model.netD( model_output.detach()).view(-1)
        losses["discriminator"] = {}
        losses["discriminator"]["outputs_fake"] = netD_fake_outputs.detach().cpu().numpy()
        losses["discriminator"]["outputs_real"] = netD_real_outputs.detach().cpu().numpy()
        losses["discriminator"]["fake"] = adversarial_criterion(netD_fake_outputs, self.fake_labels)
        losses["discriminator"]["real"] = adversarial_criterion(netD_real_outputs, self.real_labels)
        losses["discriminator"]["total"] = losses["discriminator"]["fake"] + losses["discriminator"]["real"]

        self.logger.debug('netD_real_outputs: {}'.format(netD_real_outputs))
        self.logger.debug('netD_fake_outputs: {}'.format(netD_fake_outputs))
        self.logger.debug('losses["generator"]["rec"]: {}'.format(losses["generator"]["rec"]))
        self.logger.debug('losses["generator"]["adv"]: {}'.format(losses["generator"]["adv"]))

        losses["discriminator"]["accuracy"] = self.accuracy_discriminator(losses)

        losses["discriminator"]["outputs_fake"] = np.mean(losses["discriminator"]["outputs_fake"])
        losses["discriminator"]["outputs_real"] = np.mean(losses["discriminator"]["outputs_real"])

        return losses

    def step_op(self, model, **kwargs):
        '''This function will be called every step in the training.'''
        # get inputs
        model_input = kwargs["image_{}".format(self.config["model_type"])]
        index_ = kwargs["index"]

        model_input = torch.tensor(model_input).to(self.device)


        self.logger.debug("model_input.shape: {}".format(model_input[0].shape))
        model_output = self.model(model_input)
        self.logger.debug("model_output.shape: {}".format(model_output[0].shape))
        
        # create all losses
        losses = self.criterion(model_input, model_output)
            
        def train_op():
            # This function will be executed if the model is in training mode
            if "optimization" in self.config and "reduce_lr" in self.config["optimization"]:
                # reduce the learning rate if specified
                losses["current_learning_rate"], losses["amplitude_learning_rate"] = self.update_learning_rate()
            
            # Update the generators
            self.set_requires_grad([self.model.netD], False)
            self.optimizer_G.zero_grad()
            losses["generator"]["total"].backward()
            self.optimizer_G.step()
            # Update the discriminators

            if "optimization" in self.config and "D_accuracy" in self.config["optimization"]:
                losses["discriminator"]["update"] = 0
                random_part_A = torch.rand(1)
                if (losses["discriminator"]["accuracy"] < self.config["optimization"]["D_accuracy"]) or (random_part_A < 0.01):
                    self.set_requires_grad([self.model.netD], True)
                    self.optimizer_D.zero_grad()
                    losses["discriminator"]["total"].backward()
                    self.optimizer_D.step()
                    losses["discriminator"]["update"] = 1

                self.set_requires_grad([self.model.netD], True)
            else:
                self.set_requires_grad([self.model.netD], True)
                self.optimizer_D.zero_grad()
                losses["discriminator"]["total"].backward()
                self.optimizer_D.step()
                losses["discriminator"]["update"] = 1

        def log_op():
            # This function will always execute
            logs = self.prepare_logs(losses, model_input, model_output)
            return logs

        def eval_op():
            # This function will be executed if the model is in evaluation mode
            return {}

        return {"train_op": train_op, "log_op": log_op, "eval_op": eval_op}

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

    def set_gpu(self):
        """Move the model to device cuda if available and use the specified GPU"""
        if "CUDA_VISIBLE_DEVICES" in self.config:
            if type(self.config["CUDA_VISIBLE_DEVICES"]) != str:
                self.config["CUDA_VISIBLE_DEVICES"] = str(self.config["CUDA_VISIBLE_DEVICES"])
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = self.config["CUDA_VISIBLE_DEVICES"]
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        input_img = pt2np(inputs)
        
        logs["images"].update({"batch_input": input_img})

        output_img = pt2np(predictions)

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
            
            for optimizer in [self.optimizer_G, self.optimizer_D]:  
                for g in optimizer.param_groups:
                    g['lr'] = lr
                return lr, amp
        else:
            return self.config["learning_rate"], 1

    def accuracy_discriminator(self, losses):
        with torch.no_grad():
            right_count = 0

            total_tests = 2*self.config["batch_size"]
            for i in range(self.config["batch_size"]):
                if losses["discriminator"]["outputs_real"][i] >  0.5: right_count += 1 
                if losses["discriminator"]["outputs_fake"][i] <= 0.5: right_count += 1
                
            return right_count/total_tests

    def save(self, checkpoint_path):
        '''
        'sketch_decoder' refers to the decoder of the face2sketch network
                        and vice versa. 
        '''
        state = {}
        state['encoder'] = self.model.netG.enc.state_dict()
        state['decoder'] = self.model.netG.dec.state_dict()
        state['discriminator'] = self.model.netD.state_dict()
        state['optimizer_G'] = self.optimizer_G.state_dict()
        state['optimizer_D'] = self.optimizer_D.state_dict()
        torch.save(state, checkpoint_path)

    def load(self, checkpoint_path):
        state = torch.load(checkpoint_path)
        self.model.netG.enc.load_state_dict(state['encoder'])
        self.model.netG.dec.load_state_dict(state['decoder'])
        self.model.netD.load_state_dict(state['discriminator'])
        self.optimizer_G.load_state_dict(state['optimizer_G'])
        self.optimizer_D.load_state_dict(state['optimizer_D'])

    def restore(self, checkpoint):
        state = torch.load(checkpoint)
        self.model.netG.enc.load_state_dict(state['encoder'])
        self.model.netG.dec.load_state_dict(state['decoder'])
        self.model.netD.load_state_dict(state['discriminator'])
        self.optimizer_G.load_state_dict(state['optimizer_G'])
        self.optimizer_D.load_state_dict(state['optimizer_D'])
