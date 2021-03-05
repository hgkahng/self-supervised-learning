# PyTorch Implementation of Self-Supervised Contrastive Learning Algorithms

## Requirements
- [Anaconda](https://www.anaconda.com/download/)
- [OpenCV](https://pypi.org/project/opencv-python/)
- [PyTorch](https://pytorch.org) (tested on 1.6.0)
- [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) (tested on 0.8.5)
- [Albumentations](https://github.com/albumentations-team/albumentations) (tested on 0.4.6)
```
conda update -n base conda  # use 4.8.4 or higher
conda create -n ssl python=3.8
conda activate ssl
conda install anaconda
conda install opencv -c conda-forge
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
pip install pytorch_lightning
pip install albumentations
```
