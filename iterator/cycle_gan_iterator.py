import os
import numpy as np
import random

import torch
from edflow import TemplateIterator, get_logger

import itertools
from iterator.util import (
    get_loss_funct,
    pt2np,
    set_gpu,
    set_requires_grad,
    convert_logs2numpy,
    accuracy_discriminator,
    calculate_gradient_penalty,
    update_learning_rate,
    set_random_state,
    kld_update_weight
)

###########################
###  CycleGAN Iterator  ###
###########################

class CycleGAN(TemplateIterator):
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
        # Config will be tested inside the Model class even for the iterator
        # Log the architecture of the model
        self.logger.debug(f"{model}")
        self.model = model.to(self.device)

        self.load_pretrained_vaes()

        self.optimizer_G = torch.optim.Adam(itertools.chain(self.model.netG_A.parameters(), self.model.netG_B.parameters()), lr=self.config["learning_rate"])  # betas=(opt.beta1, 0.999))
        D_lr_factor = self.config["optimization"]["D_lr_factor"] if "D_lr_factor" in config["optimization"] else 1
        self.optimizer_D_A = torch.optim.Adam(self.model.netD_A.parameters(), lr=D_lr_factor * self.config["learning_rate"])  # betas=(opt.beta1, 0.999))
        self.optimizer_D_B = torch.optim.Adam(self.model.netD_B.parameters(), lr=D_lr_factor * self.config["learning_rate"])  # betas=(opt.beta1, 0.999))

        self.add_latent_layer = bool('num_latent_layer' in self.config['variational'] and self.config['variational']['num_latent_layer'] > 0)
        self.only_latent_layer = bool('only_latent_layer' in self.config['optimization'] and self.config['optimization']['only_latent_layer'])
        if self.only_latent_layer:
            self.optimizer_Lin = torch.optim.Adam(itertools.chain(self.model.netG_A.latent_layer.parameters(), self.model.netG_B.latent_layer.parameters()), lr=self.config["learning_rate"])
            self.logger.debug("Only latent layers are optimized\nNumber of latent layers: {}".format(self.config['variational']['num_latent_layer']))
        self.real_labels = torch.ones(self.batch_size, device=self.device)
        self.fake_labels = torch.zeros(self.batch_size, device=self.device)

    def load_pretrained_vaes(self):
        log_string = "No models loaded"
        if "load_models" in self.config:
            if "sketch_path" in self.config["load_models"] and "face_path" in self.config["load_models"]:
                # load state dict of components of the VAE's
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

    def criterion(self):
        """This function returns a dictionary with all neccesary losses for the model."""

        reconstruction_criterion = get_loss_funct(self.config["losses"]["reconstruction_loss"])
        rec_weight = self.config["losses"]["reconstruction_weight"]
        adversarial_criterion = get_loss_funct(self.config["losses"]["adversarial_loss"])
        adv_weight = self.config["losses"]["adversarial_weight"]
        gp_weight = self.config["losses"]["gp_weight"] if "gp_weight" in self.config["losses"] else 0

        losses = {}

        kld_weight = self.config["losses"]["kld"]["weight"] if "kld" in self.config["losses"] else 0
        kld_delay = self.config["losses"]['kld']["delay"] if "delay" in self.config["losses"]['kld'] else 0
        kld_slope_steps = self.config["losses"]['kld']["slope_steps"] if "slope_steps" in self.config["losses"]['kld'] else 0
        losses["kld_weight"] = kld_update_weight(kld_weight, self.get_global_step(), kld_delay, kld_slope_steps)

        #########################
        ###  A: cycle sketch  ###
        #########################
        losses["sketch_cycle"] = {}
        # Generator
        losses["sketch_cycle"]["rec"] = rec_weight * torch.mean(reconstruction_criterion(self.model.output['real_A'], self.model.output['rec_A']))
        losses["sketch_cycle"]["adv"] = adv_weight * torch.mean(adversarial_criterion(self.model.netD_B(self.model.output['fake_B']).view(-1), self.real_labels))
        if kld_weight > 0 and self.model.sigma:
            losses["sketch_cycle"]["kld"] = losses["kld_weight"] * -0.5 * torch.mean(1 + self.model.netG_A.logvar - self.model.netG_A.mu.pow(2) - self.model.netG_A.logvar.exp())

        # Discriminator
        netD_B_fake_outputs = self.model.netD_B(self.model.output['fake_B'].detach()).view(-1)
        netD_B_real_outputs = self.model.netD_B(self.model.output['real_B'].detach()).view(-1)
        losses["discriminator_face"] = {}
        losses["discriminator_face"]["outputs_fake"] = netD_B_fake_outputs.clone().detach().cpu().numpy()
        losses["discriminator_face"]["outputs_real"] = netD_B_real_outputs.clone().detach().cpu().numpy()
        losses["sketch_cycle"]["disc_fake"] = adversarial_criterion(netD_B_fake_outputs, self.fake_labels)
        losses["sketch_cycle"]["disc_real"] = adversarial_criterion(netD_B_real_outputs, self.real_labels)
        losses["sketch_cycle"]["disc_total"] = losses["sketch_cycle"]["disc_fake"] + losses["sketch_cycle"]["disc_real"]
        if gp_weight > 0:
            losses["sketch_cycle"]["gp"] = gp_weight * calculate_gradient_penalty(
                discriminator=self.model.netD_B, real_images=self.model.output['real_B'], fake_images=self.model.output['fake_B'].detach(), device=self.device)
            
            losses["sketch_cycle"]["disc_total"] += losses["sketch_cycle"]["gp"]

        self.logger.debug('netD_B_real_outputs ' + str(netD_B_real_outputs))
        self.logger.debug('losses["sketch_cycle"]["disc_real"] ' + str(losses["sketch_cycle"]["disc_real"]))
        self.logger.debug('losses["sketch_cycle"]["disc_total"] ' + str(losses["sketch_cycle"]["disc_total"]))

        #######################
        ###  B: cycle face  ###
        #######################
        losses["face_cycle"] = {}
        # Generator
        losses["face_cycle"]["rec"] = rec_weight * torch.mean(reconstruction_criterion(self.model.output['real_B'], self.model.output['rec_B']))
        losses["face_cycle"]["adv"] = adv_weight * torch.mean(adversarial_criterion(self.model.netD_A(self.model.output['fake_A']).view(-1), self.real_labels))
        if kld_weight > 0 and self.model.sigma:
            losses["face_cycle"]["kld"] = losses["kld_weight"] * -0.5 * torch.mean(1 + self.model.netG_B.logvar - self.model.netG_B.mu.pow(2) - self.model.netG_B.logvar.exp())

        # Discriminator
        netD_A_fake_outputs = self.model.netD_A(self.model.output['fake_A'].detach()).view(-1)
        netD_A_real_outputs = self.model.netD_A(self.model.output['real_A'].detach()).view(-1)
        losses["discriminator_sketch"] = {}
        losses["discriminator_sketch"]["outputs_fake"] = netD_A_fake_outputs.clone().detach().cpu().numpy()
        losses["discriminator_sketch"]["outputs_real"] = netD_A_real_outputs.clone().detach().cpu().numpy()
        losses["face_cycle"]["disc_fake"] = adversarial_criterion(netD_A_fake_outputs, self.fake_labels)
        losses["face_cycle"]["disc_real"] = adversarial_criterion(netD_A_real_outputs, self.real_labels)
        losses["face_cycle"]["disc_total"] = losses["face_cycle"]["disc_fake"] + losses["face_cycle"]["disc_real"]
        if gp_weight > 0:
            losses["face_cycle"]["gp"] = gp_weight * calculate_gradient_penalty(
                discriminator=self.model.netD_A, real_images=self.model.output['real_A'], fake_images=self.model.output['fake_A'].detach(), device=self.device)
            
            losses["face_cycle"]["disc_total"] += losses["face_cycle"]["gp"]

        # Generator losses
        losses["generators"] = losses["sketch_cycle"]["rec"] + losses["face_cycle"]["rec"] + losses["sketch_cycle"]["adv"] + losses["face_cycle"]["adv"]
        if losses["kld_weight"] > 0 and self.model.sigma:
            losses["generators"] += losses["sketch_cycle"]["kld"] + losses["face_cycle"]["kld"]
        # Discriminator losses
        losses["discriminators"] = losses["sketch_cycle"]["disc_total"] + losses["face_cycle"]["disc_total"]
        # determine the accuracy of the discriminators
        losses["discriminator_sketch"]["accuracy"] = accuracy_discriminator(losses["discriminator_sketch"]["outputs_real"], losses["discriminator_sketch"]["outputs_fake"])
        losses["discriminator_face"]["accuracy"] = accuracy_discriminator(losses["discriminator_face"]["outputs_real"], losses["discriminator_face"]["outputs_fake"])

        losses["discriminator_face"]["outputs_fake"] = np.mean(losses["discriminator_face"]["outputs_fake"])
        losses["discriminator_face"]["outputs_real"] = np.mean(losses["discriminator_face"]["outputs_real"])
        losses["discriminator_sketch"]["outputs_fake"] = np.mean(losses["discriminator_sketch"]["outputs_fake"])
        losses["discriminator_sketch"]["outputs_real"] = np.mean(losses["discriminator_sketch"]["outputs_real"])

        return losses

    def step_op(self, model, **kwargs):
        '''This function will be called every step in the training.'''
        # get inputs
        real_A = kwargs["image_sketch"]
        real_B = kwargs["image_face"]

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
                D_lr_factor = self.config["optimization"]["D_lr_factor"] if "D_lr_factor" in self.config["optimization"] else 1
                losses["current_learning_rate"], losses["amplitude_learning_rate"] = update_learning_rate(
                    global_step=self.get_global_step(), num_step=self.config["num_steps"], reduce_lr=self.config["optimization"]["reduce_lr"], 
                    learning_rate=self.config["learning_rate"], list_optimizer_G=[self.optimizer_G], list_optimizer_D=[self.optimizer_D_A, self.optimizer_D_B], D_lr_factor= D_lr_factor)

            # Update the generators
            set_requires_grad([self.model.netD_A, self.model.netD_B], False)
            if not self.only_latent_layer:
                self.optimizer_G.zero_grad()
                losses["generators"].backward()
                self.optimizer_G.step()
            else:
                set_requires_grad([self.model.netG_A.enc, self.model.netG_A.dec, self.model.netG_B.enc, self.model.netG_B.dec], False)
                self.optimizer_Lin.zero_grad()
                losses["generators"].backward()
                self.optimizer_Lin.step()

            # Update the discriminators

            if "optimization" in self.config and "D_accuracy" in self.config["optimization"]:
                losses["discriminator_sketch"]["update"] = 0
                random_part_A = torch.rand(1)
                if (losses["discriminator_sketch"]["accuracy"] < self.config["optimization"]["D_accuracy"][0]) or (random_part_A < 0.01):
                    set_requires_grad([self.model.netD_A], True)
                    self.optimizer_D_A.zero_grad()
                    losses["face_cycle"]["disc_total"].backward()
                    self.optimizer_D_A.step()
                    losses["discriminator_sketch"]["update"] = 1

                losses["discriminator_face"]["update"] = 0
                random_part_B = torch.rand(1)
                if (losses["discriminator_face"]["accuracy"] < self.config["optimization"]["D_accuracy"][1]) or (random_part_B < 0.01):
                    set_requires_grad([self.model.netD_B], True)
                    self.optimizer_D_B.zero_grad()
                    losses["sketch_cycle"]["disc_total"].backward()
                    self.optimizer_D_B.step()
                    losses["discriminator_face"]["update"] = 1
                set_requires_grad([self.model.netD_A, self.model.netD_B], True)
            else:
                set_requires_grad([self.model.netD_A, self.model.netD_B], True)
                self.optimizer_D_A.zero_grad()
                self.optimizer_D_B.zero_grad()
                losses["discriminators"].backward()
                self.optimizer_D_A.step()
                self.optimizer_D_B.step()
                losses["discriminator_sketch"]["update"] = 1
                losses["discriminator_face"]["update"] = 1

        def log_op():
            # This function will always execute
            logs = self.prepare_logs(losses, [real_A, real_B], output_images)
            return logs

        def eval_op():
            # This function will be executed if the model is in evaluation mode
            return {}

        return {"train_op": train_op, "log_op": log_op, "eval_op": eval_op}

    def prepare_logs(self, losses, inputs, predictions):
        """Return a log dictionary with all instersting data to log."""
        # create a dictionary to log with all interesting variables
        logs = {
            "images": {},
            "scalars": {
                **losses
            }
        }
        ############
        ## images ##
        ############
        # input images
        real_A_img = pt2np(inputs[0])
        real_B_img = pt2np(inputs[1])
        logs["images"].update({"batch_input_sketch": real_A_img})
        logs["images"].update({"batch_input_face": real_B_img})
        # fake images
        fake_A_img = pt2np(predictions[1])
        fake_B_img = pt2np(predictions[0])
        logs["images"].update({"batch_fake_sketch": fake_A_img})
        logs["images"].update({"batch_fake_face": fake_B_img})
        # reconstructed images
        rec_A_img = pt2np(self.model.output['rec_A'])
        rec_B_img = pt2np(self.model.output['rec_B'])
        logs["images"].update({"batch_rec_sketch": rec_A_img})
        logs["images"].update({"batch_rec_face": rec_B_img})
        # log only max three images separately
        max_num = 3 if self.config["batch_size"] > 3 else self.config["batch_size"]
        for i in range(max_num):
            logs["images"].update({"input_sketch_" + str(i): np.expand_dims(real_A_img[i], 0)})
            logs["images"].update({"input_face_" + str(i): np.expand_dims(real_B_img[i], 0)})
            logs["images"].update({"fake_sketch_" + str(i): np.expand_dims(fake_A_img[i], 0)})
            logs["images"].update({"fake_face_" + str(i): np.expand_dims(fake_B_img[i], 0)})
            logs["images"].update({"rec_sketch_" + str(i): np.expand_dims(rec_A_img[i], 0)})
            logs["images"].update({"rec_face_" + str(i): np.expand_dims(rec_B_img[i], 0)})
        # convert to numpy
        logs = convert_logs2numpy(logs)
        return logs

    def save(self, checkpoint_path):
        """This function is used to save all weights of the model as well as the optimizers. 
        'sketch_decoder' refers to the decoder of the face2sketch network and vice versa.
        
        Args:
            checkpoint_path (str): Path where the weights are saved. 
        """
        state = {}
        state['sketch_encoder'] = self.model.netG_A.enc.state_dict()
        state['sketch_decoder'] = self.model.netG_B.dec.state_dict()
        state['sketch_discriminator'] = self.model.netD_A.state_dict()
        state['face_encoder'] = self.model.netG_B.enc.state_dict()
        state['face_decoder'] = self.model.netG_A.dec.state_dict()
        state['face_dicriminator'] = self.model.netD_B.state_dict()
        state['optimizer_G'] = self.optimizer_G.state_dict()
        state['optimizer_D_A'] = self.optimizer_D_A.state_dict()
        state['optimizer_D_B'] = self.optimizer_D_B.state_dict()
        if self.add_latent_layer:
            state['sketch_latent_layer'] = self.model.netG_A.latent_layer.state_dict()
            state['face_latent_layer'] = self.model.netG_B.latent_layer.state_dict()
            if self.only_latent_layer:
                state['optimizer_Lin'] = self.optimizer_Lin.state_dict()
        torch.save(state, checkpoint_path)

    def restore(self, checkpoint_path):
        """This function is used to load all weights of the model from a previous run.

        Args:
            checkpoint_path (str): Path from where the weights are loaded.
        """
        state = torch.load(checkpoint_path)
        self.model.netG_A.enc.load_state_dict(state['sketch_encoder'])
        self.model.netG_B.dec.load_state_dict(state['sketch_decoder'])
        self.model.netD_A.load_state_dict(state['sketch_discriminator'])
        self.model.netG_B.enc.load_state_dict(state['face_encoder'])
        self.model.netG_A.dec.load_state_dict(state['face_decoder'])
        self.model.netD_B.load_state_dict(state['face_discriminator'])
        self.optimizer_G.load_state_dict(state['optimizer_G'])
        self.optimizer_D_A.load_state_dict(state['optimizer_D_A'])
        self.optimizer_D_B.load_state_dict(state['optimizer_D_B'])

        if self.add_latent_layer:
            self.model.netG_A.latent_layer.load_state_dict(state['sketch_latent_layer'])
            self.model.netG_B.latent_layer.load_state_dict(state['face_latent_layer'])
            if self.only_latent_layer:
                self.optimizer_Lin.load_state_dict(state['optimizer_Lin'])

    '''
    # needed parameters: batch_size, sketch: outputs_real, outputs_fake; face: outputs_real, outputs_fake 
    def accuracy_discriminator(self, losses):
        # calculate the accuracy of the discriminators
        with torch.no_grad():
            right_count_A = 0
            right_count_B = 0

            total_tests = 2 * self.config["batch_size"]
            for i in range(self.config["batch_size"]):
                if losses["discriminator_sketch"]["outputs_real"][i] >  0.5: right_count_A += 1 
                if losses["discriminator_sketch"]["outputs_fake"][i] <= 0.5: right_count_A += 1
                
                if losses["discriminator_face"]["outputs_real"][i] >  0.5: right_count_B += 1 
                if losses["discriminator_face"]["outputs_fake"][i] <= 0.5: right_count_B += 1
                
            return right_count_A/total_tests, right_count_B/total_tests

    def _gradient_penalty(self, discriminator, real_data, generated_data):
        batch_size = real_data.size()[0]

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1, 1, 1)
        alpha = alpha.expand_as(real_data).to(self.device)
        interpolated = alpha * real_data + (1 - alpha) * generated_data
        interpolated = Variable(interpolated, requires_grad=True).to(self.device)
        # if self.use_cuda:
        #    interpolated = interpolated.cuda()

        # Calculate probability of interpolated examples
        prob_interpolated = discriminator(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size()).to(self.device),
                               create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return ((gradients_norm - 1) ** 2).mean()

    def update_learning_rate(self):
        # update learning rate does not take "D_lr_factor" into account
        step = torch.tensor(self.get_global_step(), dtype=torch.float)
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
            for optimizer in [self.optimizer_D_A, self.optimizer_D_B]:
                for g in optimizer.param_groups:
                    g['lr'] = lr*D_lr_factor
            return lr, amp
        else:
            return self.config["learning_rate"], 1

    def set_random_state(self):
        np.random.seed(self.config["random_seed"])
        torch.random.manual_seed(self.config["random_seed"])

    def kld_update(self, weight, steps, delay, slope_steps):
        output = 0
        if steps >= delay and steps < slope_steps+delay:
            output = weight*(1 - np.cos((steps-delay)*np.pi/slope_steps))/2
        if steps > slope_steps+delay:
            output = weight
        return output
    
    '''