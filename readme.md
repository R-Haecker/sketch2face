# Sketch to Face
### Project for Deep Vision lecture 2020

## Abstract
In our project the goal was to crate a translation between image domains. As concrete application we present a model to generate images of real faces from simple sketches using a modified Version of a Cycle GAN being based on a Variational Autoencoder.  
by Christopher Lüken-Winkel and Richard Häcker

For detailed information about the results have a look at our [report](assets/documents/sketch2face.pdf).
## Models
You can train and configure the following models.
- *VAE*: Variational Autoencoder
- *DCGAN*: Deep convolutional generative adversarial networks
- *VAE GAN*: generative adversarial networks with a VAE as the generator.
- *VAE WGAN*: wasserstein generative adversarial networks with a VAE as the generator.
- *Cycle GAN*
- *Cycle WGAN*

## Prerequisites
You need the following packages installed to work with our models:
* *pytorch, **edflow*
* json, yaml, numpy, matplotlib, skimage, PIL, random,

Edfow is a framework we used to train our machine learning models. You can install edflow with this command:
`pip install edflow` <br>

To train models you need to download the data sets described in the next section.

## Data sets
We used two data sets in this project with which our models will work. They are scalable with input resolutions of power of 2 and could work with any image data sets.

### Sketch
As sketch images for the training we wanted to use a dataset that contains only simple images such that everybody can produce input images that are a suitable input for the network. This lead us to the Quick Draw dataset which contains 50 million images displaying drawings over 345 categories which are created by 15 million users, each drawn in a few seconds. The category "face" which we used contains about150,000 grayscale images, each having a resolution of 28×28. We rescaled them to be of the size of 32×32. <br>

*Download* the data set as a numpy bitmap [here](https://storage.cloud.google.com/quickdraw_dataset/full/numpy_bitmap/face.npy) and save the file to: data/data_sets/full_numpy_bitmap_face.npy

### Face
The images of real faces are drawn form the Celeb A dataset. It contains around 200,00 images of the faces from about 10,000 celebrities in different orientations settings and exposure in RGB and a resolution of 178×218. The images are available in an aligned way such that a certain point of the faces is always in the same position, which we used as it implies less variation within the images simplifying the task for the network. For easier processing in the network we cropped the images around the center toget a quaratic shape and additionally rescaled the crop to either 32×32 or 64×64.

*Download* the alligned CelebA data set [here](https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?usp=drivesdk) and ectract the zip file to the folder: data/data_sets/img_align_celeba

## Train your model
To start a training run you should exectue an edflow command like the following example:

`edflow -n example_vae_wgan -b config/examples/vae_wgan.yaml -t`

With the command line argument `-n` or `-name` you can specify the name of your run. Everything important will be save at: logs/[time-stamp][name of your run]

The `-b` or `-base` argument specifies the base config for the run you are starting. This config stores all neccesary information in a `config.yaml` file.
For possible config files please have a look at these examples: config/examples/

The `-t` argument stands for training and with that edflow will start the training run.
