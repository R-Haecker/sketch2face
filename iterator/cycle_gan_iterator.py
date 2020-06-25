import os
import numpy as np
import random

import torch
import torch.nn as nn

from edflow import TemplateIterator, get_logger
from edflow.hooks.checkpoint_hooks.common import get_latest_checkpoint
from edflow.util import walk

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
        self.model = model
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["learning_rate"])# , betas=(self.config["beta1"], 0.999), weight_decay=self.config["weight_decay"])
        #self.vae.to(self.device)
    
    def set_random_state(self):
        np.random.seed(self.config["random_seed"])
        torch.random.manual_seed(self.config["random_seed"])

    def criterion(self):
        '''
        input_images: dict with keys ['image_sketch', 'image_face']
        model_output: dict: .output method of cycle_gan model CycleGAN_Model
        '''
        """This function returns a dictionary with all neccesary losses for the model."""
        reconstruction_criterion = get_loss_funct(self.config["losses"]["reconstruction_loss"])
        adversarial_criterion = get_loss_funct(self.config["losses"]["adversarial_loss"])

        real_labels = torch.ones(self.batch_size)
        fake_labels = torch.zeros(self.batch_size)

        losses = {}
        #Generator loss
        losses["rec_A"] = reconstruction_criterion(self.model.output['real_A'], self.model.output['rec_A'])
        losses["rec_B"] = reconstruction_criterion(self.model.output['real_B'], self.model.output['rec_B'])
        losses["adv_A"] = - adversarial_criterion(self.model.netD_A(self.model.output['fake_A']), fake_labels) - adversarial_criterion(self.model.netD_A(self.model.output['real_A']), real_labels)
        losses["adv_B"] = -adversarial_criterion(self.model.netD_A(self.model.output['fake_B']), fake_labels) - adversarial_criterion(self.model.netD_A(self.model.output['real_B']), real_labels)

        #Discriminator loss
        losses["disc_A"] = adversarial_criterion(self.model.netD_A(self.model.output['fake_A']).detach(), fake_labels) + adversarial_criterion(self.model.netD_A(self.model.output['real_A']).detach(), real_labels)
        losses["disc_B"] = adversarial_criterion(self.model.netD_A(self.model.output['fake_B']).detach(), fake_labels) + adversarial_criterion(self.model.netD_A(self.model.output['real_B']).detach(), real_labels)
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
            if "optimization" in self.config and "reduce_lr" in self.config["optimization"]:
                # reduce the learning rate if specified
                losses["current_learning_rate"], losses["amplitude_learning_rate"] = self.update_learning_rate()
            # This function will be executed if the model is in training mode
            for loss_key in losses.keys():
                losses[loss_key].backward(loss_key)

            self.optimizer.step()

        def log_op():
            # This function will always execute
            logs = self.prepare_logs(losses, [real_A, real_B], output_images)
            return logs

        def eval_op():
            # This function will be executed if the model is in evaluation mode
            return {}

        return {"train_op": train_op, "log_op": log_op, "eval_op": eval_op}

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
        # log the input and output images
        real_A_img = pt2np(inputs[0])
        real_B_img = pt2np(inputs[1])
        fake_B_img = pt2np(predictions[0])
        fake_A_img = pt2np(predictions[1])
        for i in range(self.config["batch_size"]):
            logs["images"].update({"real_A" + str(i): np.expand_dims(real_A_img[i],0)})
            logs["images"].update({"real_B" + str(i): np.expand_dims(real_B_img[i],0)})
            logs["images"].update({"fake_B" + str(i): np.expand_dims(fake_B_img[i],0)})
            logs["images"].update({"fake_A" + str(i): np.expand_dims(fake_A_img[i],0)})

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
            for g in self.optimizer.param_groups:
                g['lr'] = lr
            return lr, amp
        else:
            return self.config["learning_rate"], 1