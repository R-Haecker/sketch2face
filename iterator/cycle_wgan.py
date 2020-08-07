import numpy as np

import torch
from torch.autograd import Variable
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
    kld_update_weight,
    load_pretrained_vaes
)

############################################
###  Wasserstein Cycle GAN Iterator  ###
############################################

class Cycle_WGAN(TemplateIterator):
    '''This Iterator uses Wasserstein GAN's in a CycleGAN with VAE's as generators. The compatible model is the CycleWGAN_GP_VAE model.'''
    def __init__(self, config, root, model, *args, **kwargs):
        """Initialise all important parameters of the iterator."""
        super().__init__(config, root, model, *args, **kwargs)
        assert self.config["model"] == "model.cycle_gan.Cycle_WGAN", "This iterator only supports the model: model.wgan.Cycle_WGAN"
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
        self.model, log_string = load_pretrained_vaes(config=self.config, model = self.model)
        self.logger.debug(log_string)
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
            try:
                losses["sketch_cycle"] = self.losses_sketch_cycle
            except AttributeError:
                losses["sketch_cycle"] = {}
            
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
            try:
                losses["face_cycle"] = self.losses_face_cycle
            except AttributeError:
                losses["face_cycle"] = {}

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
            try:
                losses["generators"] = self.losses_generator
            except AttributeError:
                losses["generators"] = {}

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