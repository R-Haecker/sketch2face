
import numpy as np
import torch
from torch.autograd import Variable
from edflow import TemplateIterator, get_logger

from iterator.util import (
    set_gpu,
    set_random_state,
    get_loss_funct,
    update_learning_rate,
    pt2np,
    calculate_gradient_penalty,
    kld_update_weight,
    convert_logs2numpy,
    accuracy_discriminator
)
######################################
###  Wasserstein VAE GAN Iterator  ###
######################################

class VAE_WGAN(TemplateIterator):
    '''This Iterator uses Wasserstein GAN with a VAE as generator. The compatible model is the CycleWGAN_GP_VAE model.'''
    def __init__(self, config, root, model, *args, **kwargs):
        """Initialise all important parameters of the iterator."""
        super().__init__(config, root, model, *args, **kwargs)
        assert config["model_type"] != "sketch2face", "This iterator does not support sketch2face models only single GAN models supported."
        assert config["model"] == "model.vae_gan.VAE_WGAN", "This iterator only supports the model: model.vae_gan.VAE_WGAN"
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
            self.model.netG.parameters(), lr=self.learning_rate, betas=(b1, b2))
        # check if there is a different learning rate for the discriminators
        D_lr_factor = self.config["optimization"]["D_lr_factor"] if "D_lr_factor" in config["optimization"] else 1
        self.optimizer_D = torch.optim.Adam(self.model.netD.parameters(
        ), lr=self.learning_rate*D_lr_factor, betas=(b1, b2))

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
            discriminator=self.model.netD, real_images=input_images.data, fake_images=output_images.data, device=self.device)
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
        # update the generatior only every 'self.critic_iter' step
        self.update_G = bool(((self.get_global_step()) % self.critic_iter) == 0)
        
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