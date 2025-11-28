# CVPDL Homework #2 Long-Tailed Object Detection
M11315025 吳旻軒

## Model - YOLOv8

## Environment
* Kernal:Linux 6.14.0-29-generic
* OS:Ubuntu 24.04.2 LTS
* GPUs:NVIDIA GeForce RTX 4090
* torch:2.5.1(CUDA 12.1)
* torchaudio:2.5.1(CUDA 12.1)
* torchvision:0.20.1(CUDA 12.1)

## Data setting
Please download [Dataset](https://www.kaggle.com/competitions/taica-cvpdl-2025-hw-2/data) and organize them as following:
```
src/
└── datasets/
    └── images/
        ├── train/
        └── test/
```

## Installation
Use Python >= 3.8.
```bash
cd hw2_M11315025/code_M11315025
conda create -n CVPDL2 python=3.10.0
conda activate CVPDL2
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

## Training and Predict
1. Perform data preprocessing
```bash
cd src
python preprocessing.py
```
2. Training
```bash
python train.py
```
3. Predict
```bash
python predict.py
```