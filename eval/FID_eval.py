import os
import sys
import time
import yaml
import numpy as np
import torch
from torchvision.utils import save_image
from torchfid import FIDScore
from edflow.hooks.checkpoint_hooks.common import get_latest_checkpoint
# import from root directory
file_path = os.path.dirname(os.path.abspath(__file__))
working_dir = file_path[:file_path.rfind("/")]
sys.path.append(working_dir)
from model.wgan import CycleWGAN_GP_VAE
from data.data_loader.dataset import Dataset

class FID_scores():
    def __init__(self, run_name, number_samples=100):
        self.run_name = run_name
        self.r_name = run_name[20:]
        self.config, self.root, self.working_dir = self.get_config_root_wd()
        self.number_samples = number_samples
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("self.run_name",self.run_name)
        print("self.root",self.root)
        print("self.working_dir",self.working_dir)
        
        self.model = self.init_model_CycleWGAN_GP_VAE()
        self.model.eval()
        #safe_name = self.root.replace('/', '_')
        data_path = self.working_dir + "/eval/data/FID_" + str(number_samples) + self.run_name
        self.face_path = data_path + "/face"
        self.sketch_path = data_path + "/sketch"
        if not os.path.isdir(data_path):
            self.save_image_data_for_FID(data_path, number_samples)

    def save_image_data_for_FID(self, data_path, number_samples):
        data_loader = Dataset(self.config, train=False)
        
        face_path_input = self.face_path + "/input/"
        face_path_output =  self.face_path + "/output/"
            
        sketch_path_input = self.sketch_path + "/input/"
        sketch_path_output =  self.sketch_path + "/output/"

        for p in [sketch_path_input, sketch_path_output, face_path_input, face_path_output]:
            os.makedirs(p, exist_ok=False)
        with torch.no_grad():
            for i in range(number_samples):
                input_data = data_loader[i]

                input_sketch = input_data["image_sketch"]
                input_face   = input_data["image_face"]
                input_sketch = input_sketch.unsqueeze(0).to(self.device)
                input_face = input_face.unsqueeze(0).to(self.device)
                _ = self.model(real_A = input_sketch, real_B = input_face)
                output_sketch = self.model.output["rec_A"]
                output_face = self.model.output["rec_B"]
                
                save_image((input_sketch+1)/2, sketch_path_input + "image_" + str(i) + ".png")
                save_image((output_sketch+1)/2, sketch_path_output + "image_" + str(i) + ".png")
                
                save_image((input_face+1)/2, face_path_input + "image_" + str(i) + ".png")
                save_image((output_face+1)/2, face_path_output + "image_" + str(i) + ".png")
        print("done with saving")

    def cal_fid(self):
        scores = []
        self.results_path = self.get_run_results_path()
        fid_score = FIDScore(batch_size=10, verbose=True, use_cuda=True)
        for image_path in [self.sketch_path, self.face_path]:
            score = fid_score(image_path + "/input/", image_path + "/output/")
            scores.append(score)
        np.save(self.results_path + "fid_score_sketch_face.npy" ,np.asarray(scores))

        file1 = open(self.results_path + "fid_scores.txt","w") 
        L = ["FID scores of run" + self.run_name + " over " + str(self.number_samples) + " images \nsketch FID Score = " + str(scores[0]) + "\nface FID Score = " + str(scores[1]) + "\nmean FID Score = " + str((scores[0] + scores[1])/2)]  
        file1.write(L[0])
        file1.close()

    def get_config_root_wd(self):
        # initialise root, data_out and config
        def load_config(run_path):
            run_config_path = run_path + "/configs/"
            run_config_path = run_config_path + os.listdir(run_config_path)[-1]
            with open(run_config_path) as fh:
                config = yaml.full_load(fh)
            return config
        assert self.run_name[0] != "/"
        run_path = "/logs/" + self.run_name
        if run_path[-1]!=["/"]:
            run_path = run_path + "/"
        global working_dir
        root = working_dir + run_path
        config = load_config(root)
        return config, root, working_dir
        
    def get_run_results_path(self):
        a = time.time()
        timeObj = time.localtime(a)
        cur_time ='%d-%d-%d_%d-%d-%d' % (
        timeObj.tm_year, timeObj.tm_mon, timeObj.tm_mday, timeObj.tm_hour, timeObj.tm_min, timeObj.tm_sec)
        
        working_res_directory = self.working_dir + "/eval/FID"  
        results_path = working_res_directory + "/results_" + cur_time + "_" + self.r_name + "/"
        if not os.path.isdir(results_path):
            os.makedirs(results_path)
        return results_path
    
    def init_model_CycleWGAN_GP_VAE(self):
        # create all needed paths
        checkpoint_path = self.root + "train/checkpoints/" 
        # load Model with latest checkpoint
        model = CycleWGAN_GP_VAE(self.config)
        latest_chkpt_path = get_latest_checkpoint(checkpoint_root = checkpoint_path)
        model.restore(latest_chkpt_path)
        return model.to(self.device)

if __name__ == "__main__":
    run_names = [
        "2020-07-22T12-55-18_s2f_load_train_all",
        "2020-07-22T12-59-01_s2f_load_train_only_latent",
        "2020-07-21T19-04-08_cycle_wgan_from_skratch"
    ]
    #for n in run_names:
    n = run_names[2]
    f_fid = FID_scores(n, 1000)
    f_fid.cal_fid()