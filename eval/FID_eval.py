import os
import sys
import time
import json
import yaml
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image

from PIL import Image
from skimage import io

from torchfid import FIDScore
from edflow.hooks.checkpoint_hooks.common import get_latest_checkpoint
from edflow.data.believers.meta import MetaDataset

sys.path.append('/export/home/rhaecker/documents/research-of-latent-representation/GAN')
from model.gan import GAN, VAE_Model

class FID_scores():
    def __init__(self, run_name, number_samples=100):
        self.run_name = run_name
        self.root, self.config = self.init_needed_parameters(need_data_out=False)
        self.results_path = self.get_run_results_path()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.init_vae()
        
        print(self.root)
        #safe_name = self.root.replace('/', '_')
        image_path = self.root + "FID/data"
        image_path_input = image_path + "/input/"
        image_path_output =  image_path + "/output/"
        for p in [image_path, image_path_input, image_path_output]:
            if not os.path.isdir(p):
                os.makedirs(p)
                print("made directory")
        
        # get all phi loaded
        images = self.load_torch_images(number_samples, self.config["data_root"])
        
        output_images = self.model(images)
            
        for i in range(images.shape[0]):
            #print("progress ",100*i/images.shape[0])
            img = images[i]
            img.detach().cpu().numpy()
            save_image(img, image_path_input + 'image_' + str(i) + '.png')
            output_image = output_images[i]
            output_image.detach().cpu().numpy()
            save_image(output_image, image_path_output + 'image_' + str(i) + '.png')
        print("done with saving")
        
    def cal_fid(self):
        image_path = self.root + "/FID/data/"
    
        fid_score = FIDScore(batch_size=50, verbose=True, use_cuda=True)
        score = fid_score(image_path + "input/", image_path + "output/")
        print("done with fid score")
        print("score",score)
        np.save(self.results_path + "fid_score.npy" ,score)

        file1 = open(self.results_path + "fid_score.txt","w") 
        L = ["FID score of run" + self.run_name + " = " + str(score)]  
        file1.write(L[0]) 
        file1.close()

    def init_needed_parameters(self, need_data_out=True):
        # initialise root, data_out and config
        def load_config(run_path):
            run_config_path = run_path + "/configs/"
            run_config_path = run_config_path + os.listdir(run_config_path)[-1]
            with open(run_config_path) as fh:
                config = yaml.full_load(fh)
            return config

        def load_data_mem_map(run_path):
            meta_path, raw_data_path = get_raw_data_path(run_path=run_path)
            Jdata = MetaDataset(meta_path)
            return Jdata
        run_name = self.run_name
        assert run_name[0] != "/"
        prefix = "GAN/logs/"
        run_path = prefix + self.run_name
        working_directory = '/export/home/rhaecker/documents/research-of-latent-representation/'
        if run_path[-1]==["/"]:
            run_path = run_path[:-1]
        root = working_directory + run_path + "/"

        config = load_config(root)
        if need_data_out:
            data_out = load_data_mem_map(root)
            return root, data_out, config
        else:
            return root, config
        
    def get_run_results_path(self):
        a = time.time()
        timeObj = time.localtime(a)
        cur_time ='%d-%d-%d_%d-%d-%d' % (
        timeObj.tm_year, timeObj.tm_mon, timeObj.tm_mday, timeObj.tm_hour, timeObj.tm_min, timeObj.tm_sec)
        working_res_directory = '/export/home/rhaecker/documents/research-of-latent-representation/VAE/research/FID/results/'
        name_start = self.run_name.rfind("/")
        r_name = self.run_name[name_start+20:]
        results_path = working_res_directory + "/results_" + cur_time + "_" + r_name + "/"
        if not os.path.isdir(results_path):
            os.makedirs(results_path)
        return results_path
    
    def init_vae(self):
        # create all needed paths
        root = self.root
        if "/eval/" in root:
            root = root[:root.find("/eval/")]
        if root[-1] == "/":
            root = root[:-1]
        checkpoint_path = root + "/train/checkpoints/" 
        # find the name of the run
        name_start = root.rfind("/")
        run_name = root[name_start+20:]    
        # load Model with latest checkpoint
        gan = GAN(self.config)
        latest_chkpt_path = get_latest_checkpoint(checkpoint_root = checkpoint_path)
        vae = gan.generator
        vae.load_state_dict(torch.load(latest_chkpt_path)["netG"])
        vae = vae.to(self.device)
        return vae
    
    def get_path_seq_dataset(self, amount_phi, delta_phi):
        path_image_seq_dataset = "/export/home/rhaecker/documents/research-of-latent-representation/data/umap_sequences/"
        phi_range = [0,delta_phi]
        path_image_seq_dataset = path_image_seq_dataset + str(amount_phi) + "_" + str(phi_range[0]) + "_to_" + str(phi_range[1]) + "/"
        return path_image_seq_dataset

    def load_parameters(self,idx, data_root):
        # load a json file with all parameters which define the image 
        parameter_path = os.path.join(data_root, "parameters/parameters_index_" + str(idx) + ".json")
        with open(parameter_path) as f:
            parameters = json.load(f)
        return parameters

    def load_torch_images(self, number_samples, data_root):
        def load_image(idx, root):
            image_path = os.path.join(root, "images/image_index_" + str(idx) + ".png")
            image = Image.fromarray(io.imread(image_path)) 
            image = tramsform(image)
            return image

        tramsform = torchvision.transforms.Compose([transforms.Resize(size=(64,64)), transforms.ToTensor(), 
                                                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
        
        
        all_images = torch.zeros([number_samples,3,64,64]).to(self.device)
        for i in range(number_samples):
            if data_root[-7:]=="phi_seq":
                idx = np.random.randint(9999)
            else:
                idx = np.random.randint(499999)
            all_images[i] = load_image(idx, data_root)
        return all_images


if __name__ == "__main__":
    #os.environ["CUDA_VISIBLE_DEVICES"] = "9"    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    updating_run_names = ["2020-04-27T17-14-13_gan_phi_basic_red_lr_test_updating_one","2020-04-27T17-13-08_gan_phi_basic_red_lr_test_updating_one_prob","2020-04-27T17-12-01_gan_phi_basic_with_red_lr_test_updating_both","2020-04-27T17-08-10_gan_phi_basic_with_red_lr_testing_updating_acc",
                     "2020-04-27T17-05-14_gan_phi_basic_test_updating_accuracy_065","2020-04-27T17-00-54_gan_phi_basic_test_updating_both","2020-04-27T16-59-40_gan_phi_basic_test_updating_one_prob","2020-04-27T16-57-44_gan_phi_basic_updating_one"]
    te = ["2020-05-02T14-05-36_gan_big_data_phi_theta_scale_no_red_lr_lr_0001_test_updating_3","2020-05-02T14-05-26_gan_big_data_phi_theta_scale_no_red_lr_lr_0001_test_updating_2","2020-05-02T14-05-16_gan_big_data_phi_theta_scale_no_red_lr_lr_0001_test_updating_1","2020-05-02T14-05-06_gan_big_data_phi_theta_scale_no_red_lr_lr_0001_test_updating_0",
                     "2020-05-02T13-59-42_gan_big_data_phi_theta_scale__red_lr_07_lr_0001_test_updating_3","2020-05-02T13-59-32_gan_big_data_phi_theta_scale__red_lr_07_lr_0001_test_updating_2","2020-05-02T13-59-22_gan_big_data_phi_theta_scale__red_lr_07_lr_0001_test_updating_1","2020-05-02T13-59-12_gan_big_data_phi_theta_scale__red_lr_07_lr_0001_test_updating_0"]    
    
    for name in te:
        f_fid = FID_scores(name, 1000)
        f_fid.cal_fid()