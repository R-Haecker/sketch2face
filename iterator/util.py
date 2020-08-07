import os
import numpy as np
import torch
import torch.nn as nn
from torch import autograd
from torch.autograd import Variable
from models import ID_module

from edflow.util import walk

def get_loss_funct(loss_function):
    '''Get the loss function specified in the config.'''
    if loss_function == "L1":
        return nn.L1Loss()
    if loss_function == "L2" or loss_function == "MSE":
        return nn.MSELoss()
    if loss_function == "BCE":
        return nn.BCELoss()
    if loss_function == "wasserstein":
        return ID_module

# This function was copied from the VUNet repository: https://github.com/jhaux/VUNet.git
def pt2np(tensor, permute=True):
    '''Converts a torch Tensor to a numpy array.'''
    array = tensor.detach().cpu().numpy()
    if permute:
        array = np.transpose(array, (0, 2, 3, 1))
    return array


def weights_init(m):
    # custom weights initialization called on netG and netD
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def set_gpu(config):
    """Move the model to device cuda if available and use the specified GPU"""
    if "CUDA_VISIBLE_DEVICES" in config:
        if type(config["CUDA_VISIBLE_DEVICES"]) != str:
            config["CUDA_VISIBLE_DEVICES"] = str(config["CUDA_VISIBLE_DEVICES"])
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = config["CUDA_VISIBLE_DEVICES"]
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_random_state(random_seed):
    '''Set random seed for torch and numpy.'''
    np.random.seed(random_seed)
    torch.random.manual_seed(random_seed)

# this function was copied from: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/base_model.py

def set_requires_grad(nets, requires_grad=False):
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


def update_learning_rate(global_step, num_step, reduce_lr, learning_rate, list_optimizer_G, list_optimizer_D=[], D_lr_factor=1):
    # update learning rate does not take "D_lr_factor" into account
    global_step = torch.tensor(global_step, dtype=torch.float)
    current_ratio = global_step/num_step
    if current_ratio >= reduce_lr:
        def amplitide_lr(step):
            delta = (1-reduce_lr)*num_step
            return (num_step-step)/delta
        amp = amplitide_lr(global_step)
        lr = learning_rate * amp

        # Update the learning rates
        for optimizer in list_optimizer_G:
            for g in optimizer.param_groups:
                g['lr'] = lr

        for optimizer in list_optimizer_D:
            for g in optimizer.param_groups:
                g['lr'] = lr*D_lr_factor
        return lr, amp
    else:
        return learning_rate, 1


def calculate_gradient_penalty(discriminator, real_images, fake_images, device):
    '''Return the gradient penalty for the discriminator.'''
    eta = torch.FloatTensor(real_images.size()[0], 1, 1, 1).uniform_(0, 1)
    eta = eta.expand(real_images.size()[0], real_images.size(1), real_images.size(2), real_images.size(3))
    eta = eta.to(device)

    interpolated = eta * real_images + ((1 - eta) * fake_images)
    interpolated = interpolated.to(device)
    # define it to calculate gradient
    interpolated = Variable(interpolated, requires_grad=True)
    # calculate probability of interpolated examples
    prob_interpolated = discriminator(interpolated)
    # calculate gradients of probabilities with respect to examples
    gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                              grad_outputs=torch.ones(
                                  prob_interpolated.size()).to(device),
                              create_graph=True, retain_graph=True)[0]
    grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return grad_penalty


def kld_update_weight(weight, steps, delay, slope_steps):
    '''Return the amplitude of the delayed KLD loss.'''
    output = 0
    if steps >= delay and steps < slope_steps+delay:
        output = weight*(1 - np.cos((steps-delay)*np.pi/slope_steps))/2
    if steps > slope_steps+delay:
        output = weight
    return output


def convert_logs2numpy(logs):
    def conditional_convert2np(log_item):
        if isinstance(log_item, torch.Tensor):
            log_item = log_item.detach().cpu().numpy()
        return log_item
    # convert to numpy
    walk(logs, conditional_convert2np, inplace=True)
    return logs


def accuracy_discriminator(outputs_real, outputs_fake):
    with torch.no_grad():
        right_count = 0
        total_tests = 2*outputs_real.shape[0]
        for i in range(outputs_real.shape[0]):
            if outputs_real[i] > 0.5:
                right_count += 1
            if outputs_fake[i] <= 0.5:
                right_count += 1

        return right_count/total_tests

def load_pretrained_vaes(config, model):
    '''Check if pretrained models are given in the config and load them.'''
    log_string = "No models loaded"
    if "load_models" in config and config["load_models"] != None and "sketch_path" and "face_path" in config["load_models"] and config["load_models"]["sketch_path"] != None and config["load_models"]["face_path"] != None:
        # load state dict of components of the VAE's
        sketch_state = torch.load(config["load_models"]["sketch_path"])
        face_state = torch.load(config["load_models"]["face_path"])
        model.netG_A.enc.load_state_dict(sketch_state['encoder'])
        model.netG_A.dec.load_state_dict(face_state['decoder'])
        model.netD_A.load_state_dict(sketch_state['discriminator'])
        model.netG_B.enc.load_state_dict(face_state['encoder'])
        model.netG_B.dec.load_state_dict(sketch_state['decoder'])
        model.netD_B.load_state_dict(face_state['discriminator'])
        log_string = "Sketch VAE loaded from {}\nFace VAE loaded from {}".format(
            config["load_models"]["sketch_path"], config["load_models"]["face_path"])
    return model, log_string