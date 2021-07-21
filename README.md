# MasterThesis
private repo for master thesis

## Models
As the models and their weights are to large for this repo, the are not uploaded, except for the student network (PDD).

### FlowNet2
An implementation of FlowNet2 can be found [here](https://github.com/NVIDIA/flownet2-pytorch), which is a derived version from [this repo](https://github.com/ClementPinard/FlowNetPytorch). The weights used during the process, were mentioned in the second repo, and can be found [in a GDrive](https://drive.google.com/drive/folders/0B5EC7HMbyk3CbjFPb0RuODI3NmM) und the pytroch folder. As newer pytorch versions do not alsways compile the cuda kernels, in the repos above, this repo uses the pytorch implementation from [this repository](https://github.com/multimodallearning/flownet_pytorch), which is free to use.  


### File structure
The implementation of Flownet2 was put into a folder called "flownet2" into models. The whole structure was keept as is. 
The implementation of PWCNet was put into a folder called "pwc_net" into models. Here only the file structure of the subfolder PyTorch from the original git was used.
