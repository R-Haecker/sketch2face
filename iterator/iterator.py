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
        # Config will be tested inside the Model class even for the iterator
        # Log the architecture of the model
        self.logger.debug(f"{model}")
        self.vae = model
        self.optimizer = torch.optim.Adam(self.vae.parameters(), lr=self.config["learning_rate"])# , betas=(self.config["beta1"], 0.999), weight_decay=self.config["weight_decay"])
        self.vae.to(self.device)

    def set_random_state(self):
        np.random.seed(self.config["random_seed"])
        torch.random.manual_seed(self.config["random_seed"])

    def prepare_logs(self, losses, input_images, G_output_images):
        """Return a log dictionary with all instersting data to log."""
        def conditional_convert2np(log_item):
            if isinstance(log_item, torch.Tensor):
                log_item = log_item.detach().cpu().numpy()
            return log_item
        # create a dictionary to log with all interesting variables 
        # log the input and output images
        in_img = pt2np(input_images)
        out_img = pt2np(G_output_images)
        # log the images as batches
        logs["images"]["input_batch"] = in_img
        logs["images"]["output_batch"] = out_img
        # log the images separately
        for i in range(self.config["batch_size"]):
            logs["images"].update({"input_" + str(i): np.expand_dims(in_img[i],0)})
            logs["images"].update({"output_" + str(i): np.expand_dims(out_img[i],0)})
        # convert to numpy
        walk(logs, conditional_convert2np, inplace=True)
        return logs

    def criterion(self, input_images, output_images):
        """This function returns a dictionary with all neccesary losses for the model."""
        # convert to numpy
        
        recon_crit = get_loss_funct(self.config["losses"]["reconstruction_loss"])
        loss = recon_crit(input_images, output_images)

        return loss        

    def step_op(self, model, **kwargs):
        '''This function will be called every step in the training.'''
        # get inputs
        input_images = kwargs["image_face"]
        index_ = kwargs["index"]
        
        self.logger.info("input images.shape: " + str(input_images.shape))

        input_images = torch.tensor(input_images).to(self.device)

        self.logger.debug("input_images.shape: " + str(input_images.shape))
        output_images = self.vae(input_images)
        self.logger.debug("output.shape: " + str(output_images.shape))
        
        # create all losses
        loss = self.criterion(input_images, output_images)
            
        def train_op():
            # This function will be executed if the model is in training mode
            loss.backward()
            self.optimizer.step()

        def log_op():
            in_img = pt2np(input_images)
            out_img = pt2np(output_images)
            loss_np = loss.cpu().detach().numpy() 
            logs = {
                "images": {
                    "input_images": in_img,
                    "output_images": out_img,
                },
                "scalars":{
                    "L2": loss_np
                    }
            }
            return logs

        def eval_op():
            # This function will be executed if the model is in evaluation mode
            return {}

        return {"train_op": train_op, "log_op": log_op, "eval_op": eval_op}

    def save(self, checkpoint_path):
        '''Save the weights of the model to the checkpoint_path.'''
        state = {
            "model": self.vae.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }
        torch.save(state, checkpoint_path)

    def restore(self, checkpoint_path):
        '''Load the weigths of the model from a previous training.'''
        state = torch.load(checkpoint_path)
        self.vae.load_state_dict(state["vae"])
        self.optimizer.load_state_dict(state["optimizer"])
        
    def set_gpu(self):
        """Move the model to device cuda if available and use the specified GPU"""
        if "CUDA_VISIBLE_DEVICES" in self.config:
            if type(self.config["CUDA_VISIBLE_DEVICES"]) != str:
                self.config["CUDA_VISIBLE_DEVICES"] = str(self.config["CUDA_VISIBLE_DEVICES"])
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = self.config["CUDA_VISIBLE_DEVICES"]
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
