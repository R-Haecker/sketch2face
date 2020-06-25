import os
import numpy as np
import random

import torch
import torch.nn as nn

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
        
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.model.netG_A.parameters(), self.model.netG_B.parameters()), lr=self.config["learning_rate"]) # betas=(opt.beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(itertools.chain(self.model.netD_A.parameters(), self.model.netD_B.parameters()), lr=self.config["learning_rate"]) # betas=(opt.beta1, 0.999))
        
        self.real_labels = torch.ones(self.batch_size, device=self.device)
        self.fake_labels = torch.zeros(self.batch_size, device=self.device)
    
    def set_random_state(self):
        np.random.seed(self.config["random_seed"])
        torch.random.manual_seed(self.config["random_seed"])

    def criterion(self):
        """This function returns a dictionary with all neccesary losses for the model."""
        
        reconstruction_criterion = get_loss_funct(self.config["losses"]["reconstruction_loss"])
        rec_weight = self.config["losses"]["reconstruction_weight"]
        adversarial_criterion = get_loss_funct(self.config["losses"]["adversarial_loss"])
        adv_weight = self.config["losses"]["adversarial_weight"]
        losses = {}
        #########################
        ###  A: cycle sketch  ###
        #########################
        losses["sketch_cycle"] = {}
        # Generator
        losses["sketch_cycle"]["rec"] = rec_weight * torch.mean( reconstruction_criterion( self.model.output['real_A'], self.model.output['rec_A'] ) )
        losses["sketch_cycle"]["adv"] = adv_weight * torch.mean( adversarial_criterion( self.model.netD_B(self.model.output['fake_B']).view(-1) , self.real_labels ) )
        
        # Discriminator
        losses["sketch_cycle"]["disc_fake"]  = torch.mean( adversarial_criterion( self.model.netD_B( self.model.output['fake_B'].detach()).view(-1), self.fake_labels ) )  
        losses["sketch_cycle"]["disc_real"]  = torch.mean( adversarial_criterion( self.model.netD_B( self.model.output['real_B'].detach()).view(-1), self.real_labels ) )
        losses["sketch_cycle"]["disc_total"] = losses["sketch_cycle"]["disc_fake"] + losses["sketch_cycle"]["disc_real"]
        
        #######################
        ###  B: cycle face  ###
        #######################
        losses["face_cycle"] = {}
        # Generator
        losses["face_cycle"]["rec"] = rec_weight * torch.mean( reconstruction_criterion( self.model.output['real_B'], self.model.output['rec_B'] ) )
        losses["face_cycle"]["adv"] = adv_weight * torch.mean( adversarial_criterion( self.model.netD_A(self.model.output['fake_A']).view(-1), self.real_labels ) )
        
        # Discriminator
        losses["face_cycle"]["disc_fake"]  = torch.mean( adversarial_criterion( self.model.netD_A( self.model.output['fake_A'].detach()).view(-1), self.fake_labels ) )
        losses["face_cycle"]["disc_real"]  = torch.mean( adversarial_criterion( self.model.netD_A( self.model.output['real_A'].detach()).view(-1), self.real_labels ) )
        losses["face_cycle"]["disc_total"] = losses["face_cycle"]["disc_fake"] + losses["face_cycle"]["disc_real"]
        
        # Generator losses
        losses["generators"]     = losses["sketch_cycle"]["rec"] + losses["face_cycle"]["rec"] + losses["sketch_cycle"]["adv"] + losses["face_cycle"]["adv"]
        # Discriminator losses
        losses["discriminators"] = losses["sketch_cycle"]["disc_total"] + losses["face_cycle"]["disc_total"]

        return losses

    def step_op(self, model, **kwargs):
        '''This function will be called every step in the training.'''
        # get inputs
        real_A = kwargs["image_sketch"]
        real_B = kwargs["image_face"]
        index_ = kwargs["index"]

        real_A = torch.tensor(real_A).to(self.device)
        real_B = torch.tensor(real_B).to(self.device)

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
            
            # Update the generators
            self.set_requires_grad([self.model.netD_A, self.model.netD_B], False)
            self.optimizer_G.zero_grad()
            losses["generators"].backward()
            self.optimizer_G.step()
            # Update the discriminators
            self.set_requires_grad([self.model.netD_A, self.model.netD_B], True)
            self.optimizer_D.zero_grad()
            losses["discriminators"].backward()
            self.optimizer_D.step()

        def log_op():
            # This function will always execute
            logs = self.prepare_logs(losses, [real_A, real_B], output_images)
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
        state['optimizer_G'] = self.optimizer_G.state_dict()
        state['optimizer_D'] = self.optimizer_D.state_dict()
        torch.save(state, checkpoint_path)

    def load(self, checkpoint_path):
        state = torch.load(path)
        self.model.netG_A.enc.load_state_dict(state['sketch_encoder'])
        self.model.netG_B.dec.load_state_dict(state['sketch_decoder'])
        self.model.netD_A.load_state_dict(state['sketch_discriminator'])
        self.model.netG_B.enc.load_state_dict(state['face_encoder'])
        self.model.netG_A.dec.load_state_dict(state['face_decoder'])
        self.model.netD_B.load_state_dict(state['face_discriminator'])
        self.optimizer_G.load_state_dict(state['optimizer_G'])
        self.optimizer_D.load_state_dict(state['optimizer_D'])