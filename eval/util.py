import os
import yaml
import torch
# import from root directory
file_path = os.path.dirname(os.path.abspath(__file__))
working_dir = file_path[:file_path.rfind("/")]
sys.path.append(working_dir)
from model.wgan import CycleWGAN_GP_VAE

def convert_old_save_to_new(run_name, checkpoint_name):
    config_path, checkpoint_path, new_checkpoint_path = get_paths(run_name, checkpoint_name)
    with open(config_path) as file:
        config = yaml.full_load(file)
    
    model = CycleWGAN_GP_VAE(config)
    print("Model initialized")
    restore(model, checkpoint_path)
    print("Model restored from {}".format(checkpoint_path))
    save(model, new_checkpoint_path)
    print("New checkpoint saved at {}".format(new_checkpoint_path))

def get_paths(run_name, checkpoint_name):
    checkpoint_dir_path = "/export/home/rhaecker/documents/sketch2face/logs/" + run_name + "/train/checkpoints/"
    checkpoint_path = os.path.join(checkpoint_dir_path, checkpoint_name)
    
    new_checkpoint_path = os.path.join(checkpoint_dir_path, 'converted_' + checkpoint_name)
    config_path = "/export/home/rhaecker/documents/sketch2face/logs/" + run_name + "/configs"
    configs = os.listdir(config_path)
    config_path = config_path + "/" + configs[0] 
    return config_path, checkpoint_path, new_checkpoint_path

def save(model, checkpoint_path):
    state = {}
    state["encoder"] = model.netG.enc.state_dict()
    state["decoder"] = model.netG.dec.state_dict()
    state["discriminator"] = model.netD.state_dict()
    torch.save(state, checkpoint_path)
    
def restore(model, checkpoint_path):
    state = torch.load(checkpoint_path)
    model.netG.load_state_dict(state["generator"])
    model.netD.load_state_dict(state["discriminator"])

if __name__ == "__main__":
    #run_name = "2020-07-20T19-56-20_vae_wgan_big_recon_L1"
    run_name = "2020-07-20T10-30-22_vae_wgan_big_recon_L1"
    checkpoint_name = "model-50000.ckpt"
    convert_old_save_to_new(run_name, checkpoint_name)    