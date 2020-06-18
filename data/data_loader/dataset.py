import torch
import torchvision
import torchvision.transforms as transforms

from edflow import get_logger
from edflow.custom_logging import LogSingleton
from edflow.data.dataset import DatasetMixin
from edflow.util import edprint

import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import io
from PIL import Image
import json

class Dataset(DatasetMixin):
    def __init__(self, config, train=False):
        """Initialize the dataset to load training or validation images according to the config.yaml file. 
        
        :param DatasetMixin: This class inherits from this class to enable a good workflow through the framework edflow.  
        :param config: This config is loaded from the config.yaml file which specifies all neccesary hyperparameter for to desired operation which will be executed by the edflow framework.
        """
        # Create Logging for the Dataset
        if "debug_log_level" in config and config["debug_log_level"]:
            LogSingleton.set_log_level("debug")
        self.logger = get_logger("Dataset")
    
        self.data_root = self.get_data_root(config)
        # Load parameters from config
        self.config = self.set_image_res(config)
        self.set_random_state()
        
        all_indices = self.load_all_indices()
        # TODO shuffel indices choose vailadaion test and train indices. use config["shuffel_dataset"] or something 
        self.indices = all_indices
    
    def load_all_indices(self):
        # Load every indices from all images
        all_indices = [int(s[17:-5]) for s in os.listdir(self.data_root + "/parameters/")]
        return np.sort(all_indices)    

    def get_data_root(self, config):
        # Get the directory to the data
        assert "data_root" in config, "You have to specify the directory to the data in the config.yaml file."
        data_root = config["data_root"]
        if "~" in data_root:
            data_root = os.path.expanduser('~') + data_root[data_root.find("~")+1:]
        self.logger.debug("data_root: " + str(data_root))
        return data_root
    
    def set_image_res(self, config): 
        # Transforming and resizing images
        if "image_resolution" in config:    
            if type(config["image_resolution"])!=list:
                config["image_resolution"]=[config["image_resolution"], config["image_resolution"]]
            self.transform = torchvision.transforms.Compose([transforms.Resize(size=(config["image_resolution"][0],config["image_resolution"][1])), transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
            self.logger.debug("Resizing images to " + str(config["image_resolution"]))
        else:
            self.logger.info("Images will not be resized! Original image resolution will be used.")
            self.transform = torchvision.transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
        return config

    def set_random_state(self):
        if "random_seed" in self.config:
            np.random.seed(self.config["random_seed"])
            torch.random.manual_seed(self.config["random_seed"])
        else:
            self.config["random_seed"] = np.random.randint(0,2**32-1)
            np.random.seed(self.config["random_seed"])
            torch.random.manual_seed(self.config["random_seed"])

    def __len__(self):
        """This member function returns the length of the dataset
        
        :return: [description]
        :rtype: [type]
        """
        return len(self.indices)

    def get_example(self, idx):
        """This member function loads and returns images in a dictionary according to the given index.
        
        :param idx: Index of the requested image.
        :type idx: Int
        :return: Dictionary with the image at the key 'image'.
        :rtype: Dictionary
        """
        
        '''# load a json file with all parameters which define the image 
        if "request_parameters" in self.config and self.config["request_parameters"]:
            parameter_path = os.path.join(self.data_root, "parameters/parameters_index_" + str(idx) + ".json")
            with open(parameter_path) as f:
              parameters = json.load(f)
            example["parameters"] = parameters
        '''
        example = {}
        idx = self.indices[int(idx)]
        example["index"] = idx
        # Load image
        image = self.load_image(idx)
        example["image"] = image
        # Return example dictionary
        return example
        
    def load_image(self, idx):
        image_path = os.path.join(self.data_root, "images/image_index_" + str(idx) + ".png")
        image = Image.fromarray(io.imread(image_path))
        if self.transform:
            image = self.transform(image)
        return image

    def plot(self, image, name=None):
        if type(image)==dict:
            image = image["image"]
        with torch.no_grad():
            if image.shape[0]==self.config["batch_size"]:
                te = torch.zeros(self.config["batch_size"],1024,1024,3,requires_grad=False)
                te = torch.Tensor.permute(image,0,2,3,1)
                plt.figure(figsize=(20,5))
                te = (te/2+0.5).cpu()
                te.detach().numpy()
                plot_image = np.hstack(te.detach())
            else:
                te = torch.zeros(3,1024,1024,requires_grad=False)
                te = torch.Tensor.permute(image,1,2,0)
                te = (te/2+0.5).cpu()
                te.detach().numpy()
                plot_image = te
            plt.imshow(plot_image)
            if name!=None:
                path_fig=self.data_root + "/figures/first_run_150e/"
                if not os.path.isdir(path_fig):
                    os.mkdir(path_fig)
                #plt.savefig( path_fig + "figure_latent_dim_"+ str(self.latent_dim) +"_"+str(name)+".png")
            plt.show()

class DatasetTrain(Dataset):
    def __init__(self, config):
        super().__init__(config, train=True)

class DatasetEval(Dataset):
    def __init__(self, config):
        super().__init__(config, train=False)