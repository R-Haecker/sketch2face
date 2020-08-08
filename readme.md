# Sketch to Face
## Project for Deep Vision lecture 2020

## Prerequisites
You need all of the following packages installed:
* pytorch, edflow
you can install edflow via 'pip install edflow'

# training a Model
To run a training you should exectue an edflow command like the following:
`edflow -n <your run name> -b config/examples/vae_wgan.yaml -t`
With the command line argument `n` you can specify the name of your run.
With the `b` argument you specify the bas config for your in which all neccesary information for the run is stored.
For examples please have a look at the cofigs: `config/examples/`