import os
from wgan import CycleWGAN_GP_VAE
import yaml
import torch

checkpoint_dir_path = "path/to/dir"
checkpoint_name = "name.ckpt"
checkpoint_path = os.path.join(checkpoint_dir_path, checkpoint_name)
new_checkpoint_path = os.path.join(checkpoint_dir_path, 'converted_' + checkpoint_name)
config_path = "path/to/config.yaml"

def save(model, checkpoint_path):
    state = {}
    state["encoder"] = model.netG.enc.state_dict()
    state["decoder"] = model.netG.dec.state_dict()
    state["discriminator"] = model.netD.state_dict()
    state["optimizer_D"] = model.optimizer_D.state_dict()
    state["optimizer_G"] = model.optimizer_G.state_dict()

def restore(model, checkpoint_path):
    state = torch.load(checkpoint_path)
    model.netG.load_state_dict(state["generator"])
    model.netD.load_state_dict(state["discriminator"])
    model.optimizer_D.load_state_dict(state["optimizer_D"])
    model.optimizer_G.load_state_dict(state["optimizer_G"])


with open(config_path) as file:
    config = yaml.full_load(file)
    
    model = CycleWGAN_GP_VAE(config)
    print("Model initialized")

    restore(model, checkpoint_path)
    print("Model restored from {}".format(checkpoint_path))

    save(model, new_checkpoint_path)
    print("New checkpoint saved at {}".format(new_checkpoint_path))


    

