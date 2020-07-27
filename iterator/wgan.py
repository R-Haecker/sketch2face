import numpy as np
import itertools

import torch
from torch.autograd import Variable
from torch import autograd

from edflow import TemplateIterator, get_logger

from iterator.util import (
    set_gpu,
    set_random_state,
    set_requires_grad,
    get_loss_funct,
    update_learning_rate,
    pt2np,
    calculate_gradient_penalty,
    kld_update_weight,
    convert_logs2numpy
)
############################################
###  VAE Wasserstein Cycle GAN Iterator  ###
############################################


class VAE_CycleWGAN(TemplateIterator):

'''This Iterator uses Wasserstein GAN's in a CycleGAN with VAE's as generators. The compatible model is the CycleWGAN_GP_VAE model.'''
    def __init__(self, config, root, model, *args, **kwargs):
        """Initialise all important parameters of the iterator."""
        super().__init__(config, root, model, *args, **kwargs)
        assert self.config["model"] == "model.wgan.CycleWGAN_GP_VAE", "This iterator only supports the model: wgan_gradient_penalty.CycleWGAN_GP_VAE"
        # get the config and the logger
        self.config = config
        self.logger = get_logger("Iterator")
        set_random_state(self.config["random_seed"])
        # Check if cuda is available
        self.device = set_gpu(self.config)
        self.logger.debug(f"Model will pushed to the device: {self.device}")
        # Log the architecture of the model
        self.logger.debug(f"{model}")
        self.model = model.to(self.device)
        # save important constants
        self.learning_rate = self.config["learning_rate"]
        self.batch_size = self.config["batch_size"]
        self.critic_iter = self.config["losses"]["update_disc"] if "update_disc" in self.config["losses"] else 5
        # WGAN values from paper
        b1, b2 = 0.5, 0.999
        # load pretrained models if specified in the config
        self.load_pretrained_vaes()
        # check if there are latent layers
        self.add_latent_layer = bool(
            'num_latent_layer' in self.config['variational'] and self.config['variational']['num_latent_layer'] > 0)
        # check if only the latent layers should be updated
        self.only_latent_layer = bool(
            'only_latent_layer' in self.config['optimization'] and self.config['optimization']['only_latent_layer'])
        if self.only_latent_layer:
            self.critic_iter = 1
            self.logger.debug("critic_iter set to 1 since only_latent_layer is True.")
            self.optimizer_Lin = torch.optim.Adam(itertools.chain(self.model.netG_A.latent_layer.parameters(), 
                self.model.netG_B.latent_layer.parameters()), lr=self.config["learning_rate"], betas=(b1, b2))
            self.logger.debug("Only latent layers are optimized\nNumber of latent layers: {}".format(
                self.config['variational']['num_latent_layer']))
        else:
            # use ADAM optimizer
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.model.netG_A.parameters(), 
                self.model.netG_B.parameters()), lr=self.learning_rate, betas=(b1, b2))
        # check if there is a different learning rate for the discriminators
        D_lr_factor = self.config["optimization"]["D_lr_factor"] if "D_lr_factor" in config["optimization"] else 1
        self.optimizer_D = torch.optim.Adam(itertools.chain(self.model.netD_A.parameters(), 
            self.model.netD_B.parameters()), lr=D_lr_factor * self.config["learning_rate"], betas=(b1, b2))

    def load_pretrained_vaes(self):
        '''Check if pretrained models are given in the config and load them.'''
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
                log_string = "Sketch VAE loaded from {}\nFace VAE loaded from {}".format(
                    self.config["load_models"]["sketch_path"], self.config["load_models"]["face_path"])
        self.logger.debug(log_string)

    def criterion(self):
        """This function returns a dictionary with all neccesary losses for the model."""
        # get the loss criteria specified in the config
        if "sketch" in self.config["losses"]["reconstruction_loss"]:
            sketch_rec_crit = get_loss_funct(self.config["losses"]["reconstruction_loss"]["sketch"])
        if "face" in self.config["losses"]["reconstruction_loss"]:
            face_rec_crit = get_loss_funct(self.config["losses"]["reconstruction_loss"]["face"])
        else:
            face_rec_crit = sketch_rec_crit = get_loss_funct(self.config["losses"]["reconstruction_loss"])
        # get the weights for the losses specified in the config
        rec_weight = self.config["losses"]["reconstruction_weight"]
        adv_weight = self.config["losses"]["adversarial_weight"]
        gp_weight = self.config["losses"]["gp_weight"] if "gp_weight" in self.config["losses"] else 0
        kld_weight = self.config["losses"]["kld"]["weight"] if "kld" in self.config["losses"] else 0

        losses = {}
        #########################
        ###  A: cycle sketch  ###
        #########################
        ###############
        ## Generator ##
        # update the generatior only every 'self.critic_iter' step
        self.update_G = bool(((self.get_global_step()) % self.critic_iter) == 0)
        if self.update_G:
            losses["sketch_cycle"] = {}
            # the generator will be updated
            losses["update_generator"] = 1
            # reconstruction loss
            losses["sketch_cycle"]["rec"] = rec_weight * torch.mean(sketch_rec_crit(self.model.output['real_A'], self.model.output['rec_A'] ) )
            # adversarial loss
            losses["sketch_cycle"]["adv"] = adv_weight * torch.mean(self.model.netD_B(self.model.output['fake_B']).view(-1))
            # total wasserstein loss
            losses["sketch_cycle"]["generator_total"] = losses["sketch_cycle"]["rec"] - losses["sketch_cycle"]["adv"]
            # calculate the delayed KLD weight
            kld_delay = self.config["losses"]['kld']["delay"] if "delay" in self.config["losses"]['kld'] else 0
            kld_slope_steps = self.config["losses"]['kld']["slope_steps"] if "slope_steps" in self.config["losses"]['kld'] else 0
            losses["kld_weight"] = kld_update_weight(kld_weight, self.get_global_step(), kld_delay, kld_slope_steps)
            # add KLD loss if available
            if kld_weight > 0 and "sigma" in self.config["variational"] and self.config["variational"]["sigma"]:
                losses["sketch_cycle"]["kld"] = losses["kld_weight"] * -0.5 * torch.mean(
                    1 + self.model.netG_A.logvar - self.model.netG_A.mu.pow(2) - self.model.netG_A.logvar.exp())
                losses["sketch_cycle"]["generator_total"] += losses["sketch_cycle"]["kld"]
        else:
            # the generator will not be updated
            losses["update_generator"] = 0
            # log the last generator losses
            losses["sketch_cycle"] = self.losses_sketch_cycle
        ###################
        ## Discriminator ##
        losses["sketch_cycle"]["disc_fake"] = self.model.netD_B(self.model.output['fake_B'].detach()).mean()
        losses["sketch_cycle"]["disc_real"] = self.model.netD_B(self.model.output['real_B'].detach()).mean()
        losses["discriminator_face"] = {}
        # the loss of the Wasserstein GAN is the direct output of the discriminator
        losses["discriminator_face"]["outputs_fake"] = losses["sketch_cycle"]["disc_fake"].clone(
        ).detach().cpu().numpy()
        losses["discriminator_face"]["outputs_real"] = losses["sketch_cycle"]["disc_real"].clone(
        ).detach().cpu().numpy()
        # add the mean of both discriminator outputs to prevent that both outputs go to negative values
        disc_mean_output_weight = self.config["losses"]["disc_output_mean_weight"] if "disc_output_mean_weight" in self.config["losses"] else 0
        losses["sketch_cycle"]["disc_mean"]  = disc_mean_output_weight * torch.abs(torch.mean(torch.stack([losses["sketch_cycle"]["disc_fake"], losses["sketch_cycle"]["disc_real"]])))
        losses["sketch_cycle"]["disc_total"] = losses["sketch_cycle"]["disc_fake"] - losses["sketch_cycle"]["disc_real"] + losses["sketch_cycle"]["disc_mean"]
        # add gradient penalty to discriminator loss
        if gp_weight > 0:
            losses["sketch_cycle"]["gp"] = gp_weight * calculate_gradient_penalty(
                discriminator=self.model.netD_B, real_images=self.model.output['real_B'], fake_images=self.model.output['fake_B'].detach(), device=self.device)
            losses["sketch_cycle"]["disc_total"] += losses["sketch_cycle"]["gp"]
        #######################
        ###  B: cycle face  ###
        #######################
        ###############
        ## Generator ##
        # update the generatior only every 'self.critic_iter' step
        if self.update_G:
            losses["face_cycle"] = {}
            # reconstruction loss
            losses["face_cycle"]["rec"] = rec_weight * torch.mean(face_rec_crit(self.model.output['real_B'], self.model.output['rec_B'] ) )
            # adversarial loss
            losses["face_cycle"]["adv"] = adv_weight * torch.mean(self.model.netD_A(self.model.output['fake_A']).view(-1))
            # total Wasserstein loss
            losses["face_cycle"]["generator_total"] = losses["face_cycle"]["rec"] - losses["face_cycle"]["adv"]
            # add the KLD loss
            if kld_weight > 0 and self.model.sigma:
                losses["face_cycle"]["kld"] = losses["kld_weight"] * -0.5 * torch.mean(
                    1 + self.model.netG_B.logvar - self.model.netG_B.mu.pow(2) - self.model.netG_B.logvar.exp())
                losses["face_cycle"]["generator_total"] += losses["face_cycle"]["kld"]
        else:
            # the generator will not be updated
            # log the last generator losses
            losses["face_cycle"] = self.losses_face_cycle
        ###################
        ## Discriminator ##
        losses["face_cycle"]["disc_fake"] = self.model.netD_A(self.model.output['fake_A'].detach()).mean()
        losses["face_cycle"]["disc_real"] = self.model.netD_A(self.model.output['real_A'].detach()).mean()
        losses["discriminator_sketch"] = {}
        # the loss of the Wasserstein GAN is the direct output of the discriminator
        losses["discriminator_sketch"]["outputs_fake"] = losses["face_cycle"]["disc_fake"].clone(
        ).detach().cpu().numpy()
        losses["discriminator_sketch"]["outputs_real"] = losses["face_cycle"]["disc_real"].clone(
        ).detach().cpu().numpy()
        # add the mean of both discriminator outputs to prevent that both outputs go to negative values
        losses["face_cycle"]["disc_mean"]  = disc_mean_output_weight * torch.abs(torch.mean(torch.stack([losses["face_cycle"]["disc_fake"], losses["face_cycle"]["disc_real"]])))
        losses["face_cycle"]["disc_total"] = losses["face_cycle"]["disc_fake"] - losses["face_cycle"]["disc_real"] + losses["face_cycle"]["disc_mean"]
        # add gradient penalty to discriminator loss
        if gp_weight > 0:
            losses["face_cycle"]["gp"] = gp_weight * calculate_gradient_penalty(
                discriminator=self.model.netD_A, real_images=self.model.output['real_A'], fake_images=self.model.output['fake_A'].detach(), device=self.device)
            losses["face_cycle"]["disc_total"] += losses["face_cycle"]["gp"]

        # all Generator losses
        # update the generators only every 'self.critic_iter' step
        if self.update_G:
            losses["generators"] = losses["face_cycle"]["generator_total"] + losses["sketch_cycle"]["generator_total"]
            # log the losses for the steps when the generator will not be updated
            self.losses_generator = losses["generators"]
            self.losses_sketch_cycle = {x: losses["sketch_cycle"][x] for x in losses["sketch_cycle"] if x not in ["disc_fake", "disc_real", "disc_total", "gp"]}
            self.losses_face_cycle   = {x: losses["face_cycle"][x] for x in losses["face_cycle"] if x not in ["disc_fake", "disc_real", "disc_total", "gp"]}
        else:
            # set the generator loss to the last available loss
            losses["generators"] = self.losses_generator

        # all Discriminator losses
        losses["discriminators"] = losses["sketch_cycle"]["disc_total"] + losses["face_cycle"]["disc_total"]

        return losses

    def step_op(self, model, **kwargs):
        '''This funcition is executed in every training and evaluation step.'''
        # get input images
        real_A = kwargs["image_sketch"]
        real_B = kwargs["image_face"]
        real_A = Variable(torch.tensor(real_A)).to(self.device)
        real_B = Variable(torch.tensor(real_B)).to(self.device)
        # get the output images
        self.logger.debug("sketch_images.shape: " + str(real_A.shape))
        self.logger.debug("face_images.shape: " + str(real_B.shape))
        output_images = self.model(real_A, real_B)
        self.logger.debug("fake_face.shape: " + str(output_images[0].shape))
        self.logger.debug("fake_sketch.shape: " + str(output_images[1].shape))
        # create all losses
        losses = self.criterion()

        def train_op():
            '''This function will be executed if the model is in training mode'''
            # reduce the learning rate during training if specified
            if "optimization" in self.config and "reduce_lr" in self.config["optimization"]:
                optimizer_G_list = self.optimizer_Lin if self.only_latent_layer else self.optimizer_G
                D_lr_factor = self.config["optimization"]["D_lr_factor"] if "D_lr_factor" in self.config["optimization"] else 1
                losses["current_learning_rate"], losses["amplitude_learning_rate"] = update_learning_rate(
                    global_step=self.get_global_step(), num_step=self.config["num_steps"], reduce_lr=self.config["optimization"]["reduce_lr"], 
                    learning_rate=self.config["learning_rate"], list_optimizer_G=[optimizer_G_list], list_optimizer_D=[self.optimizer_D], D_lr_factor= D_lr_factor)
            # update the generators
            if self.update_G and not self.only_latent_layer:
                set_requires_grad(
                    [self.model.netD_A, self.model.netD_B], False)
                self.optimizer_G.zero_grad()
                losses["generators"].backward()
                self.optimizer_G.step()
            # train only linear layers
            if self.only_latent_layer:
                set_requires_grad([self.model.netG_A.enc, self.model.netG_A.dec,
                                   self.model.netG_B.enc, self.model.netG_B.dec], False)
                self.optimizer_Lin.zero_grad()
                losses["generators"].backward()
                self.optimizer_Lin.step()
            # update the discriminators
            set_requires_grad([self.model.netD_A, self.model.netD_B], True)
            self.optimizer_D.zero_grad()
            losses["discriminators"].backward()
            self.optimizer_D.step()

        def log_op():
            '''This function always executes and returns all logging scalars and images.'''
            logs = self.prepare_logs(losses, [real_A, real_B], output_images)
            return logs

        def eval_op():
            '''This function will be executed if the model is in evaluation mode.'''
            return {}

        return {"train_op": train_op, "log_op": log_op, "eval_op": eval_op}

    def prepare_logs(self, losses, inputs, predictions):
        """Return a log dictionary with all insteresting data to log.

        Args:
            losses (dict): A dictionary containing all important losses and skalars to log. 
            inputs (numpy.ndarray, torch.Tensor): Input images to log.
            predictions (numpy.ndarray, torch.Tensor): Output images to log.

        Returns:
            dict: A dictionary containing scalars and images in a Numpy formats.  
        """
        logs = {
            "images": {},
            "scalars": {
                **losses
            }
        }
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
        # reconstruction images
        rec_A_img = pt2np(self.model.output['rec_A'])
        rec_B_img = pt2np(self.model.output['rec_B'])
        logs["images"].update({"batch_rec_sketch": rec_A_img})
        logs["images"].update({"batch_rec_face": rec_B_img})
        # log only max three images separately
        max_num = 3 if self.batch_size > 3 else self.batch_size
        for i in range(max_num):
            logs["images"].update(
                {"input_sketch_" + str(i): np.expand_dims(real_A_img[i], 0)})
            logs["images"].update({"input_face_" + str(i): np.expand_dims(real_B_img[i], 0)})
            logs["images"].update({"fake_sketch_" + str(i): np.expand_dims(fake_A_img[i], 0)})
            logs["images"].update({"fake_face_" + str(i): np.expand_dims(fake_B_img[i], 0)})

        logs = convert_logs2numpy(logs)
        return logs

    def save(self, checkpoint_path):
        """This function is used to save all weights of the model as well as the optimizers.

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
        self.model.netD_B.load_state_dict(state['face_dicriminator'])
        self.optimizer_D.load_state_dict(state['optimizer_D'])

        if self.add_latent_layer:
            self.model.netG_A.latent_layer.load_state_dict(
                state['sketch_latent_layer'])
            self.model.netG_B.latent_layer.load_state_dict(
                state['face_latent_layer'])
        if self.only_latent_layer:
            self.optimizer_Lin.load_state_dict(state['optimizer_Lin'])
        else:
            self.optimizer_G.load_state_dict(state['optimizer_G'])

    '''
    # needed parameters: discriminator, real_images, output_images, batch_size, device, 
    def calculate_gradient_penalty(self, discriminator, real_images, output_images):
        '''
      # Return the gradient penalty for the discriminator.
      '''
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

    def kld_update(self, weight, steps, delay, slope_steps):
        '''
       # Return the amplitude of the delayed KLD loss.
       '''
        output = 0
        if steps >= delay and steps < slope_steps+delay:
            output = weight*(1 - np.cos((steps-delay)*np.pi/slope_steps))/2
        if steps >= slope_steps+delay:
            output = weight
        return output
    '''


######################################
###  Wasserstein VAE GAN Iterator  ###
######################################

class VAE_WGAN(TemplateIterator):

'''This Iterator uses Wasserstein GAN with a VAE as generator. The compatible model is the CycleWGAN_GP_VAE model.'''

    def __init__(self, config, root, model, *args, **kwargs):
        """Initialise all important parameters of the iterator."""
        super().__init__(config, root, model, *args, **kwargs)
        assert config["model_type"] != "sketch2face", "This iterator does not support sketch2face models only single GAN models supported."
        assert config["model"] == "model.wgan.CycleWGAN_GP_VAE", "This iterator only supports the model: wgan.CycleWGAN_GP_VAE"
        # get the config and the logger
        self.config = config
        self.logger = get_logger("Iterator")
        set_random_state(self.config["random_seed"])
        # Check if cuda is available
        self.device = set_gpu(self.config)
        self.logger.debug(f"Model will pushed to the device: {self.device}")
        # Log the architecture of the model
        self.logger.debug(f"{model}")
        self.model = model.to(self.device)
        # save important constants
        self.learning_rate = self.config["learning_rate"]
        self.batch_size = self.config["batch_size"]
        self.critic_iter = self.config["losses"]["update_disc"] if "update_disc" in self.config["losses"] else 5
        # WGAN values from paper
        b1, b2 = 0.5, 0.999
        # use ADAM optimizer
        self.optimizer_G = torch.optim.Adam(
            self.model.netG.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))
        # check if there is a different learning rate for the discriminators
        D_lr_factor = self.config["optimization"]["D_lr_factor"] if "D_lr_factor" in config["optimization"] else 1
        self.optimizer_D = torch.optim.Adam(self.model.netD.parameters(
        ), lr=self.learning_rate*D_lr_factor, betas=(self.b1, self.b2))

    def generate_from_sample(self):
        '''Return and generate a sample from noise.'''
        z = torch.randn(self.batch_size,
                        self.config["latent_dim"]).to(self.device)
        return self.model.netG.dec(z)

    def D_criterion(self, input_images):
        """This function calculates all losses for the discriminator.

        Args:
            input_images (torch.Tensor): Input images form the dataset.

        Returns:
            dict: dictionary containing all losses for the discriminator.
        """
        losses = {}
        losses["discriminator"] = {}
        # get the weight for the gradient penalty
        gp_weight = self.config["losses"]["gp_weight"] if "gp_weight" in self.config["losses"] else 10
        # Requires gradient for the discriminator
        for p in self.model.netD.parameters():
            p.requires_grad = True
        # discriminator output for real images
        losses["discriminator"]["real"] = self.model.netD(input_images).mean()
        # generate fake images
        output_images = self.model.netG(input_images)
        # discriminator output for fake images
        losses["discriminator"]["fake"] = self.model.netD(
            output_images.detach()).mean()
        # calculate gradient penalty
        losses["discriminator"]["gradient_penalty"] = gp_weight * calculate_gradient_penalty(
            disciminator=self.model.NetD, real_images=input_images.data, fake_images=output_images.data, device=self.device)
        # the mean of both discriminator outputs to prevent that both outputs go to negative values
        disc_mean_output_weight = self.config["losses"][
            "disc_output_mean_weight"] if "disc_output_mean_weight" in self.config["losses"] else 0
        losses["discriminator"]["mean"] = disc_mean_output_weight * torch.abs(torch.mean(torch.stack([losses["discriminator"]["fake"], losses["discriminator"]["real"]])))
        # get total discriminator loss with Wasserstein loss
        losses["discriminator"]["total"] = losses["discriminator"]["fake"] - losses["discriminator"]["real"] + losses["discriminator"]["gradient_penalty"] + losses["discriminator"]["mean"]
        losses["discriminator"]["Wasserstein_D"] = losses["discriminator"]["real"] - losses["discriminator"]["fake"]
        losses["discriminator"]["outputs_real"] = losses["discriminator"]["real"]
        losses["discriminator"]["outputs_fake"] = losses["discriminator"]["fake"]
        # generate a sample from noise if specified
        if "sample" in self.config["losses"] and self.config["losses"]["sample"]:
            self.logger.debug("Using samples for training")
            self.generated_images = self.generate_from_sample()
            losses["discriminator"]["sample"] = self.model.netD(
                self.generated_images.detach()).mean()
            # add sample loss to total discriminator loss
            losses["discriminator"]["total"] += losses["discriminator"]["sample"]
            losses["discriminator"]["outputs_sample"] = losses["discriminator"]["sample"]
        # indicate that the discriminator will be updated
        losses["discriminator"]["update"] = 1
        return losses, output_images

    def G_criterion(self, input_images):
        """This function calculates all losses for the generator.

        Args:
            input_images (torch.Tensor): The input images from the dataset.

        Returns:
            dict: dictionary containing all losses for the generator.
        """
        losses = {}
        losses["generator"] = {}
        # update the generatior only every 'self.critic_iter' step
        self.update_G = bool(((self.get_global_step()) % self.critic_iter) == 0)
        if self.update_G:
            # indicate that the generator will be updated
            losses["generator"]["update"] = 1
            # get weights of the losses
            reconstruction_criterion = get_loss_funct(self.config["losses"]["reconstruction_loss"])
            rec_weight = self.config["losses"]["reconstruction_weight"] if "reconstruction_weight" in self.config["losses"] else 1
            adv_weight = self.config["losses"]["adversarial_weight"] if "adversarial_weight" in self.config["losses"] else 1
            kld_weight = self.config["losses"]["kld"]["weight"] if "kld" in self.config["losses"] else 0
            # get the delayed KLD weight
            kld_delay = self.config["losses"]['kld']["delay"] if "delay" in self.config["losses"]['kld'] else 0
            kld_slope_steps = self.config["losses"]['kld']["slope_steps"] if "slope_steps" in self.config["losses"]['kld'] else 0
            losses["generator"]["kld_weight"] = kld_update_weight(kld_weight, self.get_global_step(), kld_delay, kld_slope_steps)
            # get KLD loss
            losses["generator"]["kld"] = 0
            if kld_weight > 0 and losses["generator"]["kld_weight"] > 0 and "sigma" in self.config["variational"] and self.config["variational"]["sigma"]:
                losses["generator"]["kld"] = losses["generator"]["kld_weight"] * -0.5 * torch.mean(
                    1 + self.model.netG.logvar - self.model.netG.mu.pow(2) - self.model.netG.logvar.exp())
            # do not compute gardients for discriminator
            for p in self.model.netD.parameters():
                p.requires_grad = False  # to avoid computation
            # generate fake images
            output_images = self.model.netG(input_images)
            # reconsturction loss
            losses["generator"]["rec"] = rec_weight * torch.mean(reconstruction_criterion(output_images, input_images))
            # adversarial loss
            losses["generator"]["adv"] = adv_weight * self.model.netD(output_images).mean()
            # total loss of generator with wasserstein loss and KLD loss
            losses["generator"]["total"] = losses["generator"]["rec"] - losses["generator"]["adv"] + losses["generator"]["kld"]
            # sample fake images from noise if specified
            if "sample" in self.config["losses"] and self.config["losses"]["sample"]:
                losses["generator"]["sample"] = self.model.netD(self.generated_images).mean()
                losses["generator"]["total"] -= adv_weight * losses["generator"]["sample"]
            # log the generator losses for steps without updating the generator
            self.losses_generator = losses["generator"]
        else:
            # get the generator losses from previous steps
            losses["generator"] = self.losses_generator
            # generator will not be updated
            losses["generator"]["update"] = 0
        return losses

    def step_op(self, model, **kwargs):
        """This funcition is executed in every training and evaluation step."""
        # get input images
        input_images = torch.from_numpy(kwargs["image_{}".format(self.config["model_type"])])
        input_images_D = Variable(input_images).to(self.device)
        input_images_G = Variable(input_images).to(self.device)
        # get discriminator losses and output images
        self.model.netD.zero_grad()
        losses, output_images = self.D_criterion(input_images_D)

        def train_op():
            '''This function will be executed if the model is in training mode'''
            # reduce the learning rate during training if specified
            if "optimization" in self.config and "reduce_lr" in self.config["optimization"]:
                D_lr_factor = self.config["optimization"]["D_lr_factor"] if "D_lr_factor" in self.config["optimization"] else 1
                losses["current_learning_rate"], losses["amplitude_learning_rate"] = update_learning_rate(
                    global_step=self.get_global_step(), num_step=self.config["num_steps"], reduce_lr=self.config["optimization"]["reduce_lr"], learning_rate=self.config["learning_rate"],
                    list_optimizer_G=[self.optimizer_G], list_optimizer_D=[self.optimizer_D], D_lr_factor= D_lr_factor)
            # Update the discriminator
            losses["discriminator"]["total"].backward()
            self.optimizer_D.step()
            # Update the generator
            if self.update_G:
                self.model.netG.zero_grad()
                g_losses = self.G_criterion(input_images_G)
                losses["generator"] = g_losses["generator"]
                losses["generator"]["total"].backward()
                self.optimizer_G.step()

        def log_op():
            '''This function always executes and returns all logging scalars and images.'''
            logs = self.prepare_logs(
                losses, input_images_D.detach(), output_images)
            return logs

        def eval_op():
            '''This function will be executed if the model is in evaluation mode.'''
            return {}

        return {"train_op": train_op, "log_op": log_op, "eval_op": eval_op}

    def prepare_logs(self, losses, input_images, output_images):
        """Return a log dictionary with all insteresting data to log.

        Args:
            losses (dict): A dictionary containing all important losses and skalars to log. 
            input_images (numpy.ndarray, torch.Tensor): Input images to log.
            output_images (numpy.ndarray, torch.Tensor): Output images to log.

        Returns:
            dict: A dictionary containing scalars and images in a Numpy formats.  
        """
        logs = {
            "images": {},
            "scalars": {
                **losses
            }
        }
        # input images
        input_img = pt2np(input_images)
        logs["images"].update({"batch_input": input_img})
        # output images
        output_img = pt2np(output_images)
        logs["images"].update({"batch_output": output_img})
        # log only max three images separately
        max_num = 3 if self.config["batch_size"] > 3 else self.config["batch_size"]
        for i in range(max_num):
            logs["images"].update({"input_" + str(i): np.expand_dims(input_img[i], 0)})
            logs["images"].update({"output_" + str(i): np.expand_dims(output_img[i], 0)})

        logs = convert_logs2numpy(logs)
        return logs

    def save(self, checkpoint_path):
        """This function is used to save all weights of the model as well as the optimizers.

        Args:
            checkpoint_path (str): Path where the weights are saved. 
        """
        state = {}
        state["encoder"] = self.model.netG.enc.state_dict()
        state["decoder"] = self.model.netG.dec.state_dict()
        state["discriminator"] = self.model.netD.state_dict()
        state["optimizer_D"] = self.optimizer_D.state_dict()
        state["optimizer_G"] = self.optimizer_G.state_dict()
        torch.save(state, checkpoint_path)
        self.logger.info('Models saved')

    def restore(self, checkpoint_path):
        """This function is used to load all weights of the model from a previous run.

        Args:
            checkpoint_path (str): Path from where the weights are loaded.
        """
        state = torch.load(checkpoint_path)
        self.model.netG.enc.load_state_dict(state["encoder"])
        self.model.netG.dec.load_state_dict(state["decoder"])
        self.model.netD.load_state_dict(state["discriminator"])
        self.optimizer_D.load_state_dict(state["optimizer_D"])
        self.optimizer_G.load_state_dict(state["optimizer_G"])

    '''
    def calculate_gradient_penalty(self, real_images, output_images):
        '''
      # Return the gradient penalty for the discriminator.
      '''
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
        prob_interpolated = self.model.netD(interpolated)
        # calculate gradients of probabilities with respect to examples
        gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(
                                   prob_interpolated.size()).to(self.device),
                               create_graph=True, retain_graph=True)[0]
        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return grad_penalty

    def kld_update(self, weight, steps, delay, slope_steps):
        '''
       # Return the amplitude of the delayed KLD loss.
       '''
        output = 0
        if steps >= delay and steps < slope_steps+delay:
            output = weight*(1 - np.cos((steps-delay)*np.pi/slope_steps))/2
        if steps > slope_steps+delay:
            output = weight
        return output
    '''

# TODO for further optimizations, extract static variables in konstruktor e.g. recon_criteria, weights for losses etc.
