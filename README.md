# GLaMur: A Gated Linear Multiscale Residual U-Net for 3D Medical Image Segmentation

## Network Design
![GLaMur Network](MICCAID.png)







<hr />

## Installation
The code is tested with PyTorch 1.11.0 and CUDA 11.3. After cloning the repository, follow the below steps for installation,

1. Install PyTorch and torchvision
```shell
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```
2. Install other dependencies
```shell
pip install -r requirements.txt
```
<hr />


## Dataset
We follow the same dataset preprocessing as in [UNETR++](https://github.com/Amshaker/unetr_plus_plus). We conducted extensive experiments on five benchmarks: Synapse, BTCV, ACDC, and Decathlon-Lung. 



 
Please refer to [Setting up the datasets](https://github.com/282857341/nnFormer) on nnFormer repository for more details.

## Training
The following scripts can be used for training our UNETR++ model on the datasets:
```shell
bash training_scripts/run_training_synapse.sh
bash training_scripts/run_training_acdc.sh
bash training_scripts/run_training_lung.sh
bash training_scripts/run_training_tumor.sh
```

<hr />

## Evaluation

The checkpoints are avilable here [!(https://drive.google.com/drive/folders/1D_yXZGsHCjAWLHMMnQKAmtKpefv-dzx3?usp=sharing)]




<hr />

## Acknowledgement
This repository is built based on [nnFormer](https://github.com/282857341/nnFormer) repository.

