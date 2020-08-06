import os
import numpy as np
import random

import torch
import torch.nn as nn

from edflow import TemplateIterator, get_logger

from iterator.util import (
    get_loss_funct,
    pt2np,
    set_gpu,
    update_learning_rate,
    set_random_state,
    convert_logs2numpy
)
#####################
### VAE Iterator  ###
#####################

class VAE(TemplateIterator):

    def __init__(self, config, root, model, *args, **kwargs):
        super().__init__(config, root, model, *args, **kwargs)
        self.logger = get_logger("Iterator")
        # export to the right gpu if specified in the config
        self.device = set_gpu(config)
        self.logger.debug(f"Model will pushed to the device: {self.device}")
        # get the config and the logger
        self.config = config
        set_random_state(random_seed=self.config["random_seed"])
        # Config will be tested inside the Model class even for the iterator
        # Log the architecture of the model
        self.logger.debug(f"{model}")
        self.vae = model
        b1, b2 = 0.5, 0.999
        self.optimizer = torch.optim.Adam(
            self.vae.parameters(), lr=self.config["learning_rate"], betas=(b1, b2))
        self.vae.to(self.device)

    def prepare_logs(self, losses, input_images, G_output_images):
        """Return a log dictionary with all instersting data to log."""
        # create a dictionary to log with all interesting variables
        # log the images as batches
        logs = {
            "images": {},
            "scalars": {
                **losses
            }
        }
        # input images
        in_img = pt2np(input_images)
        logs["images"].update({"batch_input": in_img})
        # output images
        out_img = pt2np(G_output_images)
        logs["images"].update({"batch_output": out_img})
        # log only max three images separately
        max_num = 3 if self.config["batch_size"] > 3 else self.config["batch_size"]
        for i in range(max_num):
            logs["images"].update(
                {"input_" + str(i): np.expand_dims(in_img[i], 0)})
            logs["images"].update(
                {"output_" + str(i): np.expand_dims(out_img[i], 0)})
        # convert to numpy
        logs = convert_logs2numpy(logs)
        return logs

    def criterion(self, input_images, output_images):
        """This function returns a dictionary with all neccesary losses for the model."""
        losses = {}
        reconstruction_criterion = get_loss_funct(
            self.config["losses"]["reconstruction_loss"])
        losses["reconstruction_loss"] = reconstruction_criterion(
            input_images, output_images)
        return losses

    def step_op(self, model, **kwargs):
        '''This function will be called every step in the training.'''
        # get inputs
        input_images = kwargs["image_" + self.config["model_type"]]
        input_images = torch.tensor(input_images).to(self.device)

        self.logger.debug("input_images.shape: " + str(input_images.shape))
        output_images = self.vae(input_images)
        self.logger.debug("output.shape: " + str(output_images.shape))
        # create all losses
        losses = self.criterion(input_images, output_images)

        def train_op():
            if "optimization" in self.config and "reduce_lr" in self.config["optimization"]:
                # reduce the learning rate if specified
                losses["current_learning_rate"], losses["amplitude_learning_rate"] = update_learning_rate(
                    global_step=self.get_global_step(), num_step=self.config["num_steps"],
                    reduce_lr=self.config["optimization"]["reduce_lr"],
                    learning_rate=self.config["learning_rate"], list_optimizer_G=[self.optimizer])
            # This function will be executed if the model is in training mode
            losses["reconstruction_loss"].backward()
            self.optimizer.step()

        def log_op():
            # This function will always execute
            logs = self.prepare_logs(losses, input_images, output_images)
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