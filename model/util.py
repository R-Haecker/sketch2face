import numpy as np
import torch
import torch.nn as nn

def set_random_state(config):
    np.random.seed(config["random_seed"])
    torch.random.manual_seed(config["random_seed"])
        
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