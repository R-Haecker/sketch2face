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
        self.config = config
        self.data_root_sketch, self.data_root_face = self.get_data_roots(config)
        # Load parameters from config
        self.set_image_transform(config)
        self.set_random_state()
            
        self.sketch_data = self.load_sketch_data(config)

        self.sketch_indices = self.load_sketch_indices()
        self.face_indices = self.load_face_indices()

    def load_sketch_indices(self):
        # Load indices of all sketch images
        sketch_indices = np.arange(len(self.sketch_data))
        if 'shuffle' in self.config['data'] and self.config['data']['shuffle']:
            sketch_indices = np.random.permutation(sketch_indices)
        return sketch_indices    
    
    def load_face_indices(self):
        # Load indices of all face images
        face_indices = np.asarray([s.split(".")[0] for s in os.listdir(self.data_root_face)])
        if 'shuffle' in self.config['data'] and self.config['data']['shuffle']:
            face_indices = np.random.permutation(face_indices)
        # Cut out images such that as many faces and sketches are left 
        cut_start = np.random.randint(len(face_indices)-len(self.sketch_indices))
        face_indices = face_indices[cut_start : cut_start+len(self.sketch_indices)]
        return face_indices   

    def load_sketch_data(self, config):
        #load all sketch images
        data = np.load(self.data_root_sketch)
        data = data.reshape((len(data), 28, 28))
        return data

    def get_data_roots(self, config):
        # Get the directory to the data
        data_roots = []
        for dataset in ['sketch', 'face']:
            assert "data_root_{}".format(dataset) in config['data'], "You have to specify the directory to the {} data in the config.yaml file.".format(dataset)
            data_root = config["data"]["data_root_{}".format(dataset)]
            if "~" in data_root:
                data_root = os.path.expanduser('~') + data_root[data_root.find("~")+1:]
            self.logger.debug("data_root_{}: ".format(dataset) + str(data_root))
            data_roots.append(data_root)
        return data_roots
    
    def set_image_transform(self, config):
        #mirror crop and resize depending on arguments of config
        facewidth = Image.open(os.path.join(self.data_root_face, os.listdir(self.config['data']['data_root_face'])[0])).size[0]
        transformations = [transforms.CenterCrop(facewidth)]
        if "crop_offset" in config['data']['transform']:
            transformations.append(transforms.RandomCrop(facewidth - config['data']['transform']['crop_offset']))
            self.logger.debug("Applying crops with offset {}".format(config['data']['transform']['crop_offset']))
        else: 
            self.logger.info("Images will not be cropped")
        if "mirror" in config['data']['transform'] and config['data']['transform']['mirror']:
            transformations.append(transforms.RandomHorizontalFlip())
            self.logger.debug("Applying random horizontal flip")
        else: 
            self.logger.info("Images will not be mirrored")
        if "resolution" in config['data']['transform']:
            transformations.append(transforms.Resize(config['data']['transform']['resolution']))
            self.logger.debug("Resizing imgaes to {}".format(config['data']['transform']['resolution']))
        else:
            self.logger.info("Images will not be resized")
        self.transform = transforms.Compose([*transformations, transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

    def set_random_state(self):
        if "random_seed" in self.config:
            np.random.seed(self.config["random_seed"])
            torch.random.manual_seed(self.config["random_seed"])
        else:
            self.config["random_seed"] = np.random.randint(0,2**30-1)
            np.random.seed(self.config["random_seed"])
            torch.random.manual_seed(self.config["random_seed"])

    def __len__(self):
        """This member function returns the length of the dataset
        
        :return: [description]
        :rtype: [type]
        """
        return len(self.sketch_indices)

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
        sketch_idx = self.sketch_indices[int(idx)]
        face_idx = self.face_indices[int(idx)]
        example["index"] = idx
        # Load image
        example["image_sketch"] = self.load_sketch_image(sketch_idx)
        example["image_face"] = self.load_face_image(face_idx)
        # Return example dictionary
        return example
    
    def load_sketch_image(self, idx):
        image = transforms.Normalize([0.5], [0.5])(transforms.ToTensor()(self.sketch_data[idx]))
        return image

    def load_face_image(self, idx):
        image_path = os.path.join(self.data_root_face, str(idx) + ".jpg")
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        return image

class DatasetTrain(Dataset):
    def __init__(self, config):
        super().__init__(config, train=True)

class DatasetEval(Dataset):
    def __init__(self, config):
        super().__init__(config, train=False)