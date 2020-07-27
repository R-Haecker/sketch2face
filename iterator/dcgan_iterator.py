import os
import numpy as np
import torch
import torch.nn as nn

from edflow import TemplateIterator, get_logger

from iterator.util import (
    set_gpu,
    set_random_state,
    set_requires_grad,
    get_loss_funct,
    update_learning_rate,
    pt2np,
    convert_logs2numpy,
    accuracy_discriminator
)
##########################
###  DCGAN Iterator  ###
##########################

class DCGAN(TemplateIterator):
    def __init__(self, config, root, model, *args, **kwargs):
        super().__init__(config, root, model, *args, **kwargs)
        self.logger = get_logger("Iterator")
        # export to the right gpu if specified in the config
        self.device = set_gpu(config)
        self.logger.debug(f"Model will pushed to the device: {self.device}")
        # get the config and the logger
        self.config = config
        set_random_state(self.config["random_seed"])
        self.batch_size = config['batch_size']
        # Log the architecture of the model
        self.logger.debug(f"{model}")
        self.model = model.to(self.device)

        self.optimizer_G = torch.optim.Adam(self.model.netG.parameters(), lr=self.config["learning_rate"], betas=(.5, .999))
        D_lr_factor = self.config["optimization"]["D_lr_factor"] if "D_lr_factor" in config["optimization"] else 1
        self.optimizer_D = torch.optim.Adam(self.model.netD.parameters(), lr=D_lr_factor * self.config["learning_rate"], betas=(.5, .999))

        self.real_labels = torch.ones(self.batch_size, device=self.device)
        self.fake_labels = torch.zeros(self.batch_size, device=self.device)

    def criterion(self, real_images, model_output):
        """This function returns a dictionary with all neccesary losses for the model."""
        losses = {}
        adversarial_criterion = get_loss_funct(self.config["losses"]["adversarial_loss"])

        losses["generator"] = {}
        if self.config["losses"]["adversarial_loss"] != "wasserstein":
            losses["generator"]["adv"] = torch.mean(adversarial_criterion(self.model.netD(model_output).view(-1), self.real_labels))
        else:
            losses["generator"]["adv"] = -torch.mean(self.model.netD(model_output).view(-1))

        netD_real_outputs = self.model.netD(real_images.detach()).view(-1)
        netD_fake_outputs = self.model.netD(model_output.detach()).view(-1)
        losses["discriminator"] = {}
        losses["discriminator"]["outputs_fake"] = netD_fake_outputs.detach().cpu().numpy()
        losses["discriminator"]["outputs_real"] = netD_real_outputs.detach().cpu().numpy()
        if self.config["losses"]["adversarial_loss"] != "wasserstein":
            losses["discriminator"]["fake"] = adversarial_criterion(netD_fake_outputs, self.fake_labels)
            losses["discriminator"]["real"] = adversarial_criterion(netD_real_outputs, self.real_labels)
        else:
            losses["discriminator"]["fake"] = torch.mean(netD_fake_outputs)
            losses["discriminator"]["real"] = torch.mean(netD_real_outputs)
        losses["discriminator"]["total"] = losses["discriminator"]["fake"] - losses["discriminator"]["real"]

        self.logger.debug('netD_real_outputs: {}'.format(netD_real_outputs))
        self.logger.debug('netD_fake_outputs: {}'.format(netD_fake_outputs))
        self.logger.debug('losses["generator"]["adv"]: {}'.format(losses["generator"]["adv"]))

        losses["discriminator"]["accuracy"] = accuracy_discriminator(losses["discriminator"]["outputs_real"], losses["discriminator"]["outputs_fake"])
        losses["discriminator"]["outputs_real"] = np.mean(losses["discriminator"]["outputs_real"])
        losses["discriminator"]["outputs_fake"] = np.mean(losses["discriminator"]["outputs_fake"])
        return losses

    def step_op(self, model, **kwargs):
        '''This function will be called every step in the training.'''
        # get inputs
        model_input = kwargs["random_sample"]
        real_images = kwargs["image_{}".format(self.config["model_type"])]

        model_input = torch.tensor(model_input).float().to(self.device)
        real_images = torch.tensor(real_images).float().to(self.device)

        self.logger.debug("model_input.shape: {}".format(model_input[0].shape))
        model_output = self.model(model_input)
        self.logger.debug("model_output.shape: {}".format(model_output[0].shape))
        self.wasserstein = True if self.config["losses"]['adversarial_loss'] == 'wasserstein' else False
        # create all losses
        losses = self.criterion(real_images, model_output)

        def train_op():
            # This function will be executed if the model is in training mode
            if "optimization" in self.config and "reduce_lr" in self.config["optimization"]:
                # reduce the learning rate if specified
                D_lr_factor = self.config["optimization"]["D_lr_factor"] if "D_lr_factor" in self.config["optimization"] else 1
                losses["current_learning_rate"], losses["amplitude_learning_rate"] = update_learning_rate(
                    global_step=self.get_global_step(), num_step=self.config["num_steps"], reduce_lr=self.config["optimization"]["reduce_lr"], learning_rate=self.config["learning_rate"],
                    list_optimizer_G=[self.optimizer_G], list_optimizer_D=[self.optimizer_D], D_lr_factor=D_lr_factor)

            # Update the generators
            set_requires_grad([self.model.netD], False)
            self.optimizer_G.zero_grad()
            losses["generator"]["adv"].backward()
            self.optimizer_G.step()
            # Update the discriminators

            if "optimization" in self.config and "D_accuracy" in self.config["optimization"] and not self.wasserstein:
                losses["discriminator"]["update"] = 0
                random_part_A = torch.rand(1)
                if (losses["discriminator"]["accuracy"] < self.config["optimization"]["D_accuracy"]) or (random_part_A < 0.01):
                    set_requires_grad([self.model.netD], True)
                    self.optimizer_D.zero_grad()
                    losses["discriminator"]["total"].backward()
                    self.optimizer_D.step()
                    losses["discriminator"]["update"] = 1

                set_requires_grad([self.model.netD], True)
            else:
                set_requires_grad([self.model.netD], True)
                self.optimizer_D.zero_grad()
                losses["discriminator"]["total"].backward()
                self.optimizer_D.step()
                losses["discriminator"]["update"] = 1

        def log_op():
            # This function will always execute
            logs = self.prepare_logs(losses, model_output)
            return logs

        def eval_op():
            # This function will be executed if the model is in evaluation mode
            return {}

        return {"train_op": train_op, "log_op": log_op, "eval_op": eval_op}

    def prepare_logs(self, losses, predictions):
        """Return a log dictionary with all instersting data to log."""
        # create a dictionary to log with all interesting variables
        logs = {
            "images": {},
            "scalars": {
                **losses
            }
        }
        # generated images
        output_img = pt2np(predictions)
        logs["images"].update({"batch_output": output_img})
        # log only max three images separately
        max_num = 3 if self.config["batch_size"] > 3 else self.config["batch_size"]
        for i in range(max_num):
            logs["images"].update({"output_" + str(i): np.expand_dims(output_img[i], 0)})
        # convert to numpy
        logs = convert_logs2numpy(logs)
        return logs

    def save(self, checkpoint_path):
        """This function is used to save all weights of the model as well as the optimizers. 'sketch_decoder' refers to the decoder of the face2sketch network and vice versa. 

        Args:
            checkpoint_path (str): Path where the weights are saved. 
        """
        state = {}
        state['decoder'] = self.model.netG.state_dict()
        state['discriminator'] = self.model.netD.state_dict()
        state['optimizer_G'] = self.optimizer_G.state_dict()
        state['optimizer_D'] = self.optimizer_D.state_dict()
        torch.save(state, checkpoint_path)

    def restore(self, checkpoint):
        """This function is used to load all weights of the model from a previous run.

        Args:
            checkpoint_path (str): Path from where the weights are loaded.
        """
        state = torch.load(checkpoint)
        self.model.netG.load_state_dict(state['decoder'])
        self.model.netD.load_state_dict(state['discriminator'])
        self.optimizer_G.load_state_dict(state['optimizer_G'])
        self.optimizer_D.load_state_dict(state['optimizer_D'])

    '''
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

    #needed parameters: batch_size, outputs_real, outputs_fake
    def accuracy_discriminator(self, losses):
        with torch.no_grad():
            right_count = 0

            total_tests = 2*self.config["batch_size"]
            for i in range(self.config["batch_size"]):
                if losses["discriminator"]["outputs_real"][i] >  0.5: right_count += 1 
                if losses["discriminator"]["outputs_fake"][i] <= 0.5: right_count += 1
                
            return right_count/total_tests

    '''
