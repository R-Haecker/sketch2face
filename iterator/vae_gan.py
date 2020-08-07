import numpy as np
import torch
from edflow import TemplateIterator, get_logger
import itertools

from iterator.util import (
    set_gpu,
    set_random_state,
    set_requires_grad,
    get_loss_funct,
    update_learning_rate,
    pt2np,
    calculate_gradient_penalty,
    kld_update_weight,
    convert_logs2numpy,
    accuracy_discriminator
)
##########################
###  VAE GAN Iterator  ###
##########################

class VAE_GAN(TemplateIterator):
    def __init__(self, config, root, model, *args, **kwargs):
        super().__init__(config, root, model, *args, **kwargs)
        self.logger = get_logger("Iterator")
        assert config["model"] in ["model.vae_gan.VAE_GAN", "model.vae_gan.VAE_WGAN"], "This Iterator only supports the VAE GAN models: VAE_GAN and VAE_WGAN."
        # export to the right gpu if specified in the config
        self.device = set_gpu(config)
        self.logger.debug(f"Model will pushed to the device: {self.device}")
        # get the config and the logger
        self.config = config
        set_random_state(random_seed=self.config["random_seed"])
        self.batch_size = config['batch_size']
        # Config will be tested inside the Model class even for the iterator
        # Log the architecture of the model
        self.logger.debug(f"{model}")
        self.model = model.to(self.device)

        self.optimizer_G = torch.optim.Adam(itertools.chain(self.model.netG.parameters()), lr=self.config["learning_rate"])
        D_lr_factor = self.config["optimization"]["D_lr_factor"] if "D_lr_factor" in config["optimization"] else 1
        self.optimizer_D = torch.optim.Adam(self.model.netD.parameters(), lr=D_lr_factor * self.config["learning_rate"])

        self.real_labels = torch.ones(self.batch_size, device=self.device)
        self.fake_labels = torch.zeros(self.batch_size, device=self.device)

    def criterion(self, model_input, model_output):
        """This function returns a dictionary with all neccesary losses for the model."""
        reconstruction_criterion = get_loss_funct(self.config["losses"]["reconstruction_loss"])
        adversarial_criterion = get_loss_funct(self.config["losses"]["adversarial_loss"])
        rec_weight = self.config["losses"]["reconstruction_weight"]
        adv_weight = self.config["losses"]["adversarial_weight"]
        gp_weight = self.config["losses"]["gp_weight"] if "gp_weight" in self.config["losses"] else 0
        losses = {}

        losses["generator"] = {}
        kld_weight = self.config["losses"]["kld"]["weight"] if "kld" in self.config["losses"] else 0
        kld_delay = self.config["losses"]['kld']["delay"] if "delay" in self.config["losses"]['kld'] else 0
        kld_slope_steps = self.config["losses"]['kld']["slope_steps"] if "slope_steps" in self.config["losses"]['kld'] else 0
        losses["generator"]["kld_weight"] = kld_update_weight(weight=kld_weight, steps=self.get_global_step(), delay=kld_delay, slope_steps=kld_slope_steps)

        losses["generator"]["rec"] = rec_weight * torch.mean(reconstruction_criterion(model_output, model_input))
        losses["generator"]["adv"] = adv_weight * torch.mean(adversarial_criterion(self.model.netD(model_output).view(-1), self.real_labels))
        losses["generator"]["total"] = losses["generator"]["rec"] + losses["generator"]["adv"]
        if kld_weight > 0 and self.model.sigma:
            losses["generator"]["kld"] = losses["generator"]["kld_weight"] * -0.5 * torch.mean(1 + self.model.netG.logvar - self.model.netG.mu.pow(2) - self.model.netG.logvar.exp())
            losses["generator"]["total"] += losses["generator"]["kld"]

        netD_real_outputs = self.model.netD(model_input.detach()).view(-1)
        netD_fake_outputs = self.model.netD(model_output.detach()).view(-1)
        losses["discriminator"] = {}
        losses["discriminator"]["outputs_fake"] = netD_fake_outputs.detach().cpu().numpy()
        losses["discriminator"]["outputs_real"] = netD_real_outputs.detach().cpu().numpy()
        losses["discriminator"]["fake"] = adversarial_criterion(netD_fake_outputs, self.fake_labels)
        losses["discriminator"]["real"] = adversarial_criterion(netD_real_outputs, self.real_labels)
        losses["discriminator"]["total"] = losses["discriminator"]["fake"] + losses["discriminator"]["real"]
        if gp_weight > 0:
            losses["discriminator"]["gp"] = gp_weight * calculate_gradient_penalty(discriminator=self.model.netD, real_images=model_input, fake_images=model_output.detach(), device=self.device)
            losses["discriminator"]["total"] += losses["discriminator"]["gp"]

        self.logger.debug('netD_real_outputs: {}'.format(netD_real_outputs))
        self.logger.debug('netD_fake_outputs: {}'.format(netD_fake_outputs))
        self.logger.debug('losses["generator"]["rec"]: {}'.format(losses["generator"]["rec"]))
        self.logger.debug('losses["generator"]["adv"]: {}'.format(losses["generator"]["adv"]))

        losses["discriminator"]["accuracy"] = accuracy_discriminator(losses["discriminator"]["outputs_real"], losses["discriminator"]["outputs_fake"])
        losses["discriminator"]["outputs_fake"] = np.mean(losses["discriminator"]["outputs_fake"])
        losses["discriminator"]["outputs_real"] = np.mean(losses["discriminator"]["outputs_real"])

        return losses

    def step_op(self, model, **kwargs):
        '''This function will be called every step in the training.'''
        # get inputs
        model_input = kwargs["image_{}".format(self.config["model_type"])]

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
                D_lr_factor = self.config["optimization"]["D_lr_factor"] if "D_lr_factor" in self.config["optimization"] else 1
                losses["current_learning_rate"], losses["amplitude_learning_rate"] = update_learning_rate(
                    global_step=self.get_global_step(), num_step=self.config["num_steps"], reduce_lr=self.config["optimization"]["reduce_lr"], learning_rate=self.config["learning_rate"],
                    list_optimizer_G=[self.optimizer_G], list_optimizer_D=[self.optimizer_D], D_lr_factor=D_lr_factor)

            # Update the generators
            set_requires_grad([self.model.netD], False)
            self.optimizer_G.zero_grad()
            losses["generator"]["total"].backward()
            self.optimizer_G.step()

            # Update the discriminators
            if "optimization" in self.config and "D_accuracy" in self.config["optimization"]:
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
            logs = self.prepare_logs(losses, model_input, model_output)
            return logs

        def eval_op():
            # This function will be executed if the model is in evaluation mode
            return {}

        return {"train_op": train_op, "log_op": log_op, "eval_op": eval_op}

    def prepare_logs(self, losses, inputs, predictions):
        """Return a log dictionary with all instersting data to log."""
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
        input_img = pt2np(inputs)
        logs["images"].update({"batch_input": input_img})
        # output images
        output_img = pt2np(predictions)
        logs["images"].update({"batch_output": output_img})
        # log only max three images separately
        max_num = 3 if self.config["batch_size"] > 3 else self.config["batch_size"]
        for i in range(max_num):
            logs["images"].update({"input_" + str(i): np.expand_dims(input_img[i], 0)})
            logs["images"].update({"output_" + str(i): np.expand_dims(output_img[i], 0)})
        # convert to numpy
        logs = convert_logs2numpy(logs)
        return logs

    def save(self, checkpoint_path):
        """This function is used to save all weights of the model as well as the optimizers.'sketch_decoder' refers to the decoder of the face2sketch network and vice versa. 

        Args:
            checkpoint_path (str): Path where the weights are saved. 
        """
        state = {}
        state['encoder'] = self.model.netG.enc.state_dict()
        state['decoder'] = self.model.netG.dec.state_dict()
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
        self.model.netG.enc.load_state_dict(state['encoder'])
        self.model.netG.dec.load_state_dict(state['decoder'])
        self.model.netD.load_state_dict(state['discriminator'])
        self.optimizer_G.load_state_dict(state['optimizer_G'])
        self.optimizer_D.load_state_dict(state['optimizer_D'])