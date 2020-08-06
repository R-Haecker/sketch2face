import torch
import torch.nn as nn
from edflow import get_logger
import numpy as np
import torch.utils.data
from model.modules import Transpose2dBlock, ExtraConvBlock, Conv2dBlock, LinLayers
from torch.autograd import Variable

###################
###  VAE Model  ###
###################

class VAE(nn.Module):
    """This Variational Autoencoder is scalable and able to be changed to suit the different requirements."""

    def __init__(self,
                 latent_dim,
                 min_channels,
                 max_channels,
                 in_size,
                 in_channels,
                 out_size,
                 out_channels,
                 sigma=False,
                 num_extra_conv_enc=0,
                 num_extra_conv_dec=0,
                 block_activation=nn.ReLU(),
                 final_activation=nn.Tanh(),
                 batch_norm_enc=True,
                 batch_norm_dec=True,
                 drop_rate_enc=None,
                 drop_rate_dec=None,
                 bias_enc=True,
                 bias_dec=True,
                 same_max_channels=False,
                 num_latent_layer=0):
        """This constructor enables to create a custom varaitional Autoencoder.

        Args:
            latent_dim (int): The dimensionality of the latent space.
            min_channels (int): Channel dimension after the first convolution is applied.
            max_channels (int): Channel dimension is double after every convolutional block up to the value 'max_channels'
            in_size (int): Spatial resolution of the input image.
            in_channels (int): Channel dimension of the input image.
            out_size (int): Spatial resolution of the output image.
            out_channels (int): Channel dimension of the output image.
            sigma (bool, optional): If the encoder will predict the standard deviation of the normal distribution in the latent space. Defaults to False.
            num_extra_conv_enc (int, optional): Number of additional convolutions in the encoder which preserve the spatial size in every convolutional blocks. Defaults to 0.
            num_extra_conv_dec (int, optional): Number of additional convolutions in the decoder which preserve the spatial size in every convolutional blocks. Defaults to 0.
            block_activation (torch.nn module, optional): Activation function used in the convolutional blocks. Defaults to nn.ReLU().
            final_activation (torch.nn module, optional): Activation function used for the output image. Defaults to nn.Tanh().
            batch_norm_enc (bool, optional): Normalize over the batch size in the encoder. Defaults to True.
            batch_norm_dec (bool, optional): Normalize over the batch size in the decoder. Defaults to True.
            drop_rate_enc (float, optional): Dropout rate for the convolutions in the encoder. Defaults to None, corresponding to no dropout.
            drop_rate_dec (float, optional): Dropout rate for the convolutions in the decoder. Defaults to None, corresponding to no dropout.
            bias_enc (bool, optional): If the convolutions in the encoder use a bias. Defaults to True.
            bias_dec (bool, optional): If the convolutions in the decoder use a bias. Defaults to True.
            num_latent_layer (int, optional): Number of linear layers in the latent space. They use the sampled latent representation as an input and feed the output to the decoder. Defaults to 0.
        """
        # TODO same_max_channels doc_string
        super(VAE, self).__init__()
        # log all parameters
        self.logger = get_logger("VAE_Model")
        self.logger.debug("latent_dim: {}".format(latent_dim))
        self.logger.debug("min_channels: {}".format(min_channels))
        self.logger.debug("max_channels: {}".format(max_channels))
        self.logger.debug("in_size: {}".format(in_size))
        self.logger.debug("in_channels: {}".format(in_channels))
        self.logger.debug("out_size: {}".format(out_size))
        self.logger.debug("out_channels: {}".format(out_channels))
        self.logger.debug("sigma: {}".format(sigma))
        self.logger.debug("num_extra_conv_enc: {}".format(num_extra_conv_enc))
        self.logger.debug("num_extra_conv_dec: {}".format(num_extra_conv_dec))
        self.logger.debug("block_activation: {}".format(block_activation))
        self.logger.debug("final_activation: {}".format(final_activation))
        self.logger.debug("batch_norm_enc: {}".format(batch_norm_enc))
        self.logger.debug("batch_norm_dec: {}".format(batch_norm_dec))
        self.logger.debug("drop_rate_enc: {}".format(drop_rate_enc))
        self.logger.debug("drop_rate_dec: {}".format(drop_rate_dec))
        self.logger.debug("bias_enc: {}".format(bias_enc))
        self.logger.debug("bias_dec: {}".format(bias_dec))
        self.logger.debug("same_max_channels: {}".format(same_max_channels))
        self.logger.debug("num_latent_layer: {}".format(num_latent_layer))
        enc_min_channels = dec_min_channels = min_channels
        enc_max_channels = dec_max_channels = max_channels
        if same_max_channels:
            power = np.log2(in_size)
            enc_min_channels = int(2**(power - (np.log2(out_size) - np.log2(in_size)) - 2))
            self.logger.info("Adjusted min enc_min_channels to {} for max_channels of encoder and decoder to match.".format(enc_min_channels))
        # set corresponding bottleneck
        self.bottleneck = self.bottleneck_sigma if sigma else self.bottleneck_no_sigma
        # initialize the encoder and the decoder
        self.enc = VAE_Encoder(latent_dim, enc_min_channels, enc_max_channels, in_size, in_channels, sigma, num_extra_conv_enc, block_activation, batch_norm_enc, drop_rate_enc, bias_enc)
        self.dec = VAE_Decoder(latent_dim, dec_min_channels, dec_max_channels, out_size, out_channels, num_extra_conv_dec, block_activation, final_activation, batch_norm_dec, drop_rate_dec, bias_dec)
        # add linear latent layers if specified
        self.add_linear_layers = False
        if num_latent_layer > 0:
            self.add_linear_layers = True
            self.latent_layer = LinLayers(num_latent_layer, latent_dim, sigma)
            self.logger.info("Added {} linear layers layers".format(num_latent_layer))

    def bottleneck_sigma(self, x):
        """This function connects the encoder to the decoder if sigma is predicted.

        Args:
            x (torch.Tensor): Input tensor containing the mean and standard deviation for the normal distribution.

        Returns:
            torch.Tensor: Return a sample from the normal distribution for the decoder.
        """
        self.mu = x[:, :x.shape[-1]//2]
        self.logvar = x[:, x.shape[-1]//2:]
        self.std = self.logvar.mul(0.5).exp_()
        self.logger.debug("varitaional mu.shape: " + str(self.mu.shape))
        self.logger.debug("varitaional var.shape: " + str(self.std.shape))
        eps = Variable(self.std.data.new(self.std.size()).normal_())
        # final latent representation
        x = self.mu + self.std * eps
        return x

    def bottleneck_no_sigma(self, x):
        """This function connects the encoder to the decoder if no sigma is predicted.

        Args:
            x (torch.Tensor): Input tensor containing the mean for the normal distribution.

        Returns:
            torch.Tensor: Return a sample from the normal distribution for the decoder.
        """
        self.mu = x
        self.std = torch.ones_like(x)
        self.logger.debug("varitaional mu.shape: " + str(self.mu.shape))
        eps = Variable(self.std.data.new(self.std.size()).normal_())
        # final latent representation
        x = self.mu + self.std * eps
        return x

    def forward(self, x):
        """This function generates reconstructions of the input images.

        Args:
            x (torch.Tensor): Tensor of input images with the shape: [b, c, h, w]; with b = batch dimension, c = input channels, h = input height, w = input width.

        Returns:
            torch.Tensor: Reconstructed images.
        """
        x = self.enc(x)
        if self.add_linear_layers:
            x = self.latent_layer(x)
        x = x.reshape(x.shape[:-2])
        x = self.bottleneck(x)
        x = self.dec(x)
        return x

###########################
###  VAE Encoder Model  ###
###########################

class VAE_Encoder(nn.Module):
    """This encoder is part of the VAE model class."""

    def __init__(self,
                 latent_dim,
                 min_channels,
                 max_channels,
                 in_size,
                 in_channels,
                 sigma=False,
                 num_extra_conv=0,
                 block_activation=nn.ReLU(),
                 batch_norm=True,
                 drop_rate=None,
                 bias=True):
        """This is the constructor for a custom encoder.

        Args:
            latent_dim (int): The dimensionality of the latent space.
            min_channels (int): Channel dimension after the first convolution is applied.
            max_channels (int): Channel dimension is double after every convolutional block up to the value 'max_channels'.
            in_size (int): Spatial resolution of the input image.
            in_channels (int): Channel dimension of the input image.
            sigma (bool, optional): If we predict the standard deviation of the normal distribution in the latent space. Defaults to False.
            num_extra_conv (int, optional): Number of additional convolutions in every convolutional blocks preserving the spatial size. Defaults to 0.
            block_activation (torch.nn module, optional): Activation function used in the convolutional blocks. Defaults to nn.ReLU().
            batch_norm (bool, optional): Normalize over the batch size. Defaults to True.
            drop_rate (float, optional): Dropout rate for the convolutions. Defaults to None, corresponding to no dropout.
            bias (bool, optional): If the convolutions use a bias. Defaults to True.
        """
        super(VAE_Encoder, self).__init__()
        self.logger = get_logger("Encoder")
        # create a list with all channel dimensions throughout the encoder.
        layers = []
        latent_dim = 2*latent_dim if sigma else latent_dim
        channel_numbers = [in_channels] + list(np.minimum(min_channels * 2**np.arange(np.log2(in_size)).astype(np.int), np.ones(int(np.log2(in_size)), dtype=int) * int(max_channels))) + [latent_dim]
        # get all convolutional blocks with corresponding parameters
        for i in range(len(channel_numbers)-1):
            in_ch = channel_numbers[i]
            out_ch = channel_numbers[i+1]
            # add numb of extra convolutions
            for i in range(num_extra_conv):
                layers.append(ExtraConvBlock(in_ch, block_activation, batch_norm, drop_rate, bias))
            # add convolution
            if i != len(channel_numbers)-2:
                layers.append(Conv2dBlock(in_ch, out_ch, block_activation, batch_norm, drop_rate, bias))
            else:
                layers.append(Conv2dBlock(in_ch, out_ch, block_activation, batch_norm, drop_rate, bias, stride=1))
        # save all blocks to the class instance
        self.main = nn.Sequential(*layers)
        self.logger.debug("Encoder channel sizes: {}".format(channel_numbers))

    def forward(self, x):
        """This function predicts the properties of a normal distribution."""
        return self.main(x)

###########################
###  VAE Decoder Model  ###
###########################

class VAE_Decoder(nn.Module):
    """This decoder is part of the VAE model."""

    def __init__(self,
                 latent_dim,
                 min_channels,
                 max_channels,
                 out_size,
                 out_channels,
                 num_extra_conv=0,
                 block_activation=nn.ReLU(),
                 final_activation=nn.Tanh(),
                 batch_norm=True,
                 drop_rate=None,
                 bias=True):
        """This is the constructor for a custom decoder.

        Args:
            latent_dim (int): The dimensionality of the latent space.
            min_channels (int): Channel dimension before the last convolution is applied.
            max_channels (int): Channel dimension after the first convolution is applied. The channel dimension is cut in half after every convolutional block.
            out_size (int): Spatial resolution of the output image.
            out_channels (int): Channel dimension of the output image.
            num_extra_conv (int, optional): Number of additional convolutions in every convolutional blocks preserving the spatial size. Defaults to 0.
            block_activation (torch.nn module, optional): Activation function used in the convolutional blocks. Defaults to nn.ReLU().
            final_activation (torch.nn module, optional): Activation function used in the last convolution for the output image. Defaults to nn.Tanh().
            batch_norm (bool, optional): Normalize over the batch size. Defaults to True.
            drop_rate (float, optional): Dropout rate for the convolutions. Defaults to None, corresponding to no dropout.
            bias (bool, optional): If the convolutions use a bias. Defaults to True.
        """
        super(VAE_Decoder, self).__init__()
        self.logger = get_logger("Decoder")
        # create a list with all channel dimensions throughout the decoder.
        layers = []
        channel_numbers = [latent_dim] + list(np.minimum(min_channels * 2**np.arange(np.log2(out_size)-2)[::-1].astype(np.int), np.ones(int(np.log2(out_size)-2), dtype=int) * int(max_channels)))
        # get all convolutional blocks with corresponding parameters
        for i in range(len(channel_numbers)-1):
            in_ch = channel_numbers[i]
            out_ch = channel_numbers[i+1]
            stride = 2 if i != 0 else 1
            padding = 1 if i != 0 else 0
            layers.append(Transpose2dBlock(in_ch, out_ch, block_activation, batch_norm, drop_rate, bias, stride=stride, padding=padding))
            for i in range(num_extra_conv):
                layers.append(ExtraConvBlock(out_ch, block_activation, batch_norm, drop_rate, bias))

        if num_extra_conv > 0:
            layers.append(Transpose2dBlock(min_channels, out_channels, block_activation,  batch_norm, drop_rate, bias))
        else:
            layers.append(Transpose2dBlock(min_channels, out_channels, final_activation, batch_norm=False, drop_rate=None, bias=bias))
        for i in range(num_extra_conv):
            if i != num_extra_conv-1:
                layers.append(ExtraConvBlock(out_channels, block_activation, batch_norm, drop_rate, bias))
            else:
                layers.append(ExtraConvBlock(out_channels, final_activation, batch_norm=False, drop_rate=None, bias=bias))
        # save all blocks to the class instance
        self.main = nn.Sequential(*layers)
        self.logger.debug("Decoder channel sizes: {}".format(channel_numbers + [out_channels]))

    def forward(self, x):
        """This function creates reconstructed image from latent representations."""
        x = x[..., None, None]
        return self.main(x)

#########################
### VAE config Model  ###
#########################

class VAE_config(VAE):
    """This class inherits from the :VAE:class: class to parse all parameters from a given config to the VAE class."""

    def __init__(self, config):
        """This constructor parases the parameters from the config to the :VAE:class:.

        Args:
            config (dict): This dictionary is loaded from a yaml file which contains all necessary information for a training run.

        Config Keyword Args:
        :key model_type:  
        :type model_type: str
        :param conv: 
        :type conv: dict

        config: `dict`
        Config dictionary with keys:

        ``"model_type"``
            This (`str`) parameter can be either 'sketch' or 'face'. It decides which input data will be used and the corresponding parameters for the model will be parsed to the :VAE:class:.

        ``"conv"``
            This dictionary (`dict`) contains the information for the convolutional operations in the :VAE:class:.

            - ``"n_channel_start"``, (`int`) Number of channels after first convolution. 

            - ``"n_channel_max"``, (`int`) Number of channels double after every convolutional block until this maximum channel dimension is reached.

            - ``"sketch_extra_conv"``, (`int`) Number of additional convolutions for the "sketch" model in every convolutional block which preserve the spatial dimension. 

            - ``"face_extra_conv"``, (`int`) Number of additional convolutions for the "face" model in every convolutional block which preserve the spatial dimension. 

        ``"variational"``
            This dictionary (`dict`) contains the information for the architecture in the :VAE:class:.

            - ``"sigma"``, (`bool`) If ``True`` the VAE encoder will predict the standard deviation of the normal distribution in the latent space. 

            - ``"num_latent_layer"``, (`int`) Number of linear layers in the bottleneck.

        ``"batch_norm"``
            This Flag (`bool`) normalize the data over the batch size.

        ``"dropout"``
            - ``"enc_rate"``, (`float`) specifies the rate of dropout for the encoder.

            - ``"dec_rate"``, (`float`) specifies the rate of dropout for the decoder.

        ``"bias"``
            - ``"enc"``, (`bool`) If ``True`` the convolution operations in the encoder will have a bias offset parameter.

            - ``"dec"``, (`bool`) If ``True`` the convolution operations in the decoder will have a bias offset parameter.  
        """
        self.config = config
        # extract all parameters for the VAE model from the config
        if self.config["model_type"] == "sketch":
            num_extra_conv_enc = num_extra_conv_dec = self.config['conv']['sketch_extra_conv'] if "sketch_extra_conv" in self.config["conv"] else 0
            in_size = out_size = 32
            in_channels = out_channels = 1
        elif self.config["model_type"] == "face":
            num_extra_conv_enc = num_extra_conv_dec = self.config['conv']['face_extra_conv'] if "face_extra_conv" in self.config["conv"] else 0
            in_size = out_size = self.config['data']['transform']['resolution']
            in_channels = out_channels = 3
        else:
            assert(0), "ERROR: wrong model_type in config. With this VAE model you can only use the model_type:'sketch' or 'face'."
        latent_dim = self.config["latent_dim"]
        min_channels = self.config['conv']['n_channel_start']
        max_channels = self.config['conv']['n_channel_max']
        sigma = self.config['variational']['sigma']
        block_activation = nn.ReLU()
        final_activation = nn.Tanh()
        batch_norm_enc = batch_norm_dec = self.config['batch_norm']
        drop_rate_enc = self.config['dropout']['enc_rate'] if "dropout" in self.config and "enc_rate" in self.config["dropout"] else 0
        drop_rate_dec = self.config['dropout']['dec_rate'] if "dropout" in self.config and "dec_rate" in self.config["dropout"] else 0
        bias_enc = self.config['bias']['enc'] if "bias" in self.config and "enc" in self.config["bias"] else True
        bias_dec = self.config['bias']['dec'] if "bias" in self.config and "dec" in self.config["bias"] else True
        same_max_channels = False
        num_latent_layer = self.config['variational']['num_latent_layer'] if 'num_latent_layer' in self.config['variational'] else 0
        # initialise the VAE model
        super(VAE_config, self).__init__(
            latent_dim,
            min_channels,
            max_channels,
            in_size,
            in_channels,
            out_size,
            out_channels,
            sigma,
            num_extra_conv_enc,
            num_extra_conv_dec,
            block_activation,
            final_activation,
            batch_norm_enc,
            batch_norm_dec,
            drop_rate_enc,
            drop_rate_dec,
            bias_enc,
            bias_dec,
            same_max_channels,
            num_latent_layer)
