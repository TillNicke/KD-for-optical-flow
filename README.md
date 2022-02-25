# MasterThesis
Private repo for master thesis of Till Nicke. Thesis topic: "Distilling knowledge for real-time image registration in ultrasound sequences of veins and arteries"

# File structure
The folders contain the following files

## Models
Flownet2 without weights can be found under models/flownet_pytorch/ . The Architecture was kept as found [here](https://github.com/NVIDIA/flownet2-pytorch). Modifications were made regarding the correlation layer, channel norm, and resampling layers. PyTorch implementations can be found [here](https://github.com/multimodallearning/flownet_pytorch/blob/main/flownet2_components.py).
The same implementations were used for the PWC-Net, which can be found under models/pwc_net/. The original implementation was forked from the official [repo](https://github.com/NVlabs/PWC-Net).
The implementation of the PDD-Net, which is barely used in the notebooks, but still is available, can be found under models/pdd_net .

## Utils
In the utils folder, functions that are used for vizualisation, model loading, encoding, etc. can be found.

## Abdominal Experiments
The abdominal experiments are a line of experiments, that were conducted before the access to ultrasound data was available. They served as a sanity check and prove of concept for the actualy experiments

## Data
Notebooks on how the data is pre-processed, and the creation of two distinct data sets. The data is not available for distribution and therefore not included.

# Experiments
Experiment 1 contains the training of a PDD-Net on reference data. 

Experiment 2 contains knowledge distillation experiments using Flownet2, and/or  PWC-Net as teachers

Experiment 3 contains deep mutual learning experiments.

 
