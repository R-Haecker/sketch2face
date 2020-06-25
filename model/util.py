import numpy as np
import torch
import torch.nn as nn

def get_tensor_shapes(config, sktech, encoder = True):
    """This function calculates the shape of a every tensor after an operation in the VAE_Model.        
    :return: A list of the shape of an tensors after every module.
    :rtype: List
    """        
    tensor_shapes = []
    # The first shape is specified in the config
    first_channel = 1 if sketch else 3
    resolution = config["data"]["transform"]["resolution"] if sketch else 32
    tensor_shapes.append([first_channel, resolution, resolution])
    # how many downsampling blow will we need
    n_blocks = int(np.round(np.log2(resolution)))
    if sketch:
        extra_conv = bool("sketch_extra_conv" in config["conv"] and config["conv"]["sketch_extra_conv"] != 0)
    else:
        extra_conv = bool("face_extra_conv" in config["conv"] and config["conv"]["face_extra_conv"] != 0)
    if extra_conv:
        tensor_shapes.append([config["conv"]["n_channel_start"], tensor_shapes[0][-1], tensor_shapes[0][-1]])
        range_ = [1, n_blocks+1]
    else:
        range_ = [0, n_blocks]
    
    # calculate the shape after a convolutuonal operation
    for i in range(*range_):
        spacial_res = tensor_shapes[i][-1]//2
        if i==0:
            channels = config["conv"]["n_channel_start"]
        else:
            channels = np.minimum(tensor_shapes[i][0]*2, config["conv"]["n_channel_max"])
        
        if (encoder and i == n_blocks - 1 and not extra_conv and "variational" in config and "sigma" in config["variational"] and config["variational"]["sigma"]) or (encoder and i == n_blocks and extra_conv and "variational" in config and "sigma" in config["variational"] and config["variational"]["sigma"]):
            # if variational with sigma we want to out put double the channel dim at last operation for mu and sigma
            tensor_shapes.append([channels * 2, spacial_res, spacial_res])
        else:
            tensor_shapes.append([channels, spacial_res, spacial_res])
    return tensor_shapes

def set_random_state(config):
    np.random.seed(config["random_seed"])
    torch.random.manual_seed(config["random_seed"])

def test_config(config):
    ''' Test the config if it will work with the VAE_Model.'''
    assert "random_seed" in config, "If you use the normla data set there should be a 'random_seed' in the config."
    # Check for all neccesary information about the data 
    assert "data" in config, "We need to specify data properties"
    assert "data_root_sketch" in config["data"]
    assert "data_root_face" in config["data"]
    assert "validation_split" in config["data"]
    assert "shuffle" in config["data"]
    assert "transform" in config["data"]
    assert "resolution" in config["data"]["transform"]
    assert type(config["data"]["transform"]["resolution"]) == int, "Only use square face images with given int resolution"
    # ensure right model type
    assert "model_type" in config, "We have to specify a model type"
    assert config["model_type"] in ["sketch2face", "face", "sketch"], "Choose from given model types"

    assert "activation_function" in config, "For this model you need to specify the activation function: possible options :{'ReLU, LeakyReLu, Sigmoid, LogSigmoid, Tanh, SoftMax'}"
    # check for infromation about convolutions
    assert "conv" in config, "You have to use convolutional operations specified in config['conv']"
    assert "n_channel_start" in config["conv"], "We need to specify with how many channels we start"
    assert "n_channel_max" in config["conv"], "We need to specify how many channels we want to end with"
    
    # Test config for iterator parameters
    assert "losses" in config, "You have to specify the losses used in the model in config['losses']"
    assert "reconstruction_loss" in config["losses"], "The config must contain and define a Loss function for image reconstruction. possibilities:{'L1','L2'or'MSE'}."
    assert "learning_rate" in config, "The config must contain and define a the learning rate."
    
def get_act_func(config, logger):
    """This function retruns the specified activation function from the config."""

    if config["activation_function"] == "ReLU":
        if "ReLU" in config:
            logger.debug("activation function: changed ReLu to leakyReLU with secified slope!")
            return nn.LeakyReLU(negative_slope=config["ReLu"])
        else:
            logger.debug("activation function: ReLu")
            return nn.ReLU(True)  
    if config["activation_function"] == "LeakyReLU":
        if "LeakyReLU_negative_slope" in config:
            logger.debug("activation_function: LeakyReLU")
            return nn.LeakyReLU(negative_slope=config["LeakyReLU_negative_slope"])
        elif "LeakyReLU" in config:
            logger.debug("activation_function: LeakyReLU")
            return nn.LeakyReLU(negative_slope=config["LeakyReLU"])
        else:
            logger.debug("activation function: LeakyReLu changed to ReLU because no slope value could be found")
            return nn.LeakyReLU()
    if config["activation_function"] == "Sigmoid":
        logger.debug("activation_function: Sigmoid")
        return nn.Sigmoid
    if config["activation_function"] == "LogSigmoid":
        logger.debug("activation_function: LogSigmoid")
        return nn.LogSigmoid
    if config["activation_function"] == "Tanh":
        logger.debug("activation_function: Tanh")
        return nn.Tanh
    if config["activation_function"] == "SoftMax":
        logger.debug("activation_function: SoftMax")
        return nn.SoftMax()