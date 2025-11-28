# CVPDL Homework #3 Image Generation
M11315025 吳旻軒

## Model - Denoising Diffusion Probabilistic Models

## Environment
* Kernal:Linux 6.14.0-29-generic
* OS:Ubuntu 24.04.2 LTS
* GPUs:NVIDIA GeForce RTX 4090
* torch:2.5.1(CUDA 12.1)
* torchaudio:2.5.1(CUDA 12.1)
* torchvision:0.20.1(CUDA 12.1)

## Data setting
Please download [Dataset](https://drive.google.com/file/d/1xVCJD6M6sE-tZJYxenLzvuEkSiYXig_F/view) and organize them as following:
```
src/
└── datasets/ <---- place 60000 images here
```

## Installation
Use Python >= 3.10.
```bash
cd hw3_M11315025
conda create -n CVPDL3 python=3.10.0
conda activate CVPDL3
```
Installation of required packages:
```bash
pip install -r requirements.txt
```
If you want to use the GPU. Install PyTorch/CUDA dependent packages (depends on your CUDA) :
```bash
pip uninstall torch torchaudio torchvision
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

## Training and Generation
1. Training
```bash
cd src
python train.py
```
2. Generation
```bash
python inference.py
```
3. FID Implementation

Use pytorch-fid to calculate FID:
```bash
pip install pytorch-fid
```
Calculate FID with the training dataset
```bash
python -m pytorch_fid ./img_M11315025 ./datasets
```
Calculate FID with precalculated mean and covariance. mnist.npz: downloaded from [link](https://drive.google.com/file/d/1QQMFWsdcCyD1HfCnwvIgrbcPPICDChCG/view) and place in the /src folder.
```bash
python -m pytorch_fid ./img_M11315025 ./mnist.npz
```