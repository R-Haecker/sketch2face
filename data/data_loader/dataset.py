import json
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
import yaml

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
        self.config = config
        self.data_types = self.setup_data_types()
        
        self.data_roots = self.get_data_roots()
        # Load parameters from config
        self.set_image_transforms()
        self.set_random_state()
        
        # Yet a bit sloppy but ok
        self.sketch_data = self.load_sketch_data()

        self.indices = self.load_indices(train)

    def setup_data_types(self):
        # Reads from model type which data is needed
        model_type = self.config['model_type']
        data_types = []
        if 'sketch' in model_type:
            data_types.append('sketch')
        if 'face' in model_type:
            data_types.append('face')
        return data_types

    def load_indices(self, train):
        # Load indices of all sketch images
        shuffle = 'shuffle' in self.config['data'] and self.config['data']['shuffle']
        indices = {}
        for data_type in self.data_types:
            if data_type == 'sketch':
                type_indices = np.arange(len(self.sketch_data))
                num_sketches = len(type_indices)
                if shuffle:
                    type_indices = np.random.permutation(type_indices)
            if data_type == 'face':
                type_indices = np.asarray([s.split(".")[0] for s in os.listdir(self.data_roots['face'])])
                if shuffle:
                    type_indices = np.random.permutation(type_indices)
                if 'sketch' in self.data_types:
                    cut_start = np.random.randint(len(type_indices)-num_sketches)
                    type_indices = type_indices[cut_start : cut_start+num_sketches]
            if train:
                type_indices = type_indices[:int(len(type_indices)*(1-self.config['data']['validation_split']))]
            else:
                type_indices = type_indices[int(len(type_indices)*(1-self.config['data']['validation_split'])):]
            indices[data_type] = type_indices
        return indices
    
    def load_sketch_data(self):
        #load all sketch images
        if 'sketch' in self.data_types:
            data = np.load(self.data_roots['sketch'])
            data = data.reshape((len(data), 28, 28, 1))
            return data
        else:
            return

    def get_data_roots(self):
        # Get the directory to the data
        data_roots = {}
        for data_type in self.data_types:
            assert "data_root_{}".format(data_type) in self.config['data'], "You have to specify the directory to the {} data in the config.yaml file.".format(data_type)
            data_root = self.config["data"]["data_root_{}".format(data_type)]
            if "~" in data_root:
                data_root = os.path.expanduser('~') + data_root[data_root.find("~")+1:]
            self.logger.debug("data_root_{}: ".format(data_type) + str(data_root))
            data_roots[data_type] = data_root
        return data_roots
    
    def set_image_transforms(self):
        #mirror crop and resize depending on arguments of config
        self.transforms = {}
        for data_type in self.data_types:
            if data_type == 'sketch':
                transformations = transforms.Compose([transforms.ToPILImage(), transforms.Resize(32), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
            
            if data_type == 'face':
                facewidth = Image.open(os.path.join(self.data_roots['face'], os.listdir(self.data_roots['face'])[0])).size[0]
                transformations = [transforms.CenterCrop(facewidth)]
                if "crop_offset" in self.config['data']['transform']:
                    transformations.append(transforms.RandomCrop(facewidth - self.config['data']['transform']['crop_offset']))
                    self.logger.debug("Applying crops with offset {}".format(self.config['data']['transform']['crop_offset']))
                else: 
                    self.logger.info("Images will not be cropped")
                if "mirror" in self.config['data']['transform'] and self.config['data']['transform']['mirror']:
                    transformations.append(transforms.RandomHorizontalFlip())
                    self.logger.debug("Applying random horizontal flip")
                else: 
                    self.logger.info("Images will not be mirrored")
                if "resolution" in self.config['data']['transform']:
                    transformations.append(transforms.Resize(self.config['data']['transform']['resolution']))
                    self.logger.debug("Resizing imgaes to {}".format(self.config['data']['transform']['resolution']))
                else:
                    self.logger.info("Images will not be resized")
                transformations = transforms.Compose([*transformations, transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
            self.transforms[data_type] = transformations

    def set_random_state(self):
        if "random_seed" in self.config:
            np.random.seed(self.config["random_seed"])
            torch.random.manual_seed(self.config["random_seed"])
        else:
            raise ValueError("Enter random_seed in config.")

    def __len__(self):
        """This member function returns the length of the dataset
        
        :return: [description]
        :rtype: [type]
        """
        return len(self.indices[self.data_types[0]])

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
        example["index"] = idx
        for data_type in self.data_types:
            type_idx = self.indices[data_type][int(idx)]
            # Load image
            example["image_{}".format(data_type)] = self.load_image(type_idx, data_type)
        # Return example dictionary
        return example
    
    def load_image(self, idx, data_type):
        if data_type == 'sketch':
            image = self.sketch_data[idx]
        if data_type == 'face':
            image_path = os.path.join(self.data_roots['face'], str(idx) + ".jpg")
            image = Image.open(image_path)
        image = self.transforms[data_type](image)
        return image

class DatasetTrain(Dataset):
    def __init__(self, config):
        super().__init__(config, train=True)

class DatasetEval(Dataset):
    def __init__(self, config):
        super().__init__(config, train=False)