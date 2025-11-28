# CVPDL Homework #1 Object Detection
M11315025 吳旻軒

## Model - YOLO11

## Environment
* Kernal:Linux 6.14.0-29-generic
* OS:Ubuntu 24.04.2 LTS
* GPUs:NVIDIA GeForce RTX 4090
* torch:2.5.1(CUDA 12.1)
* torchaudio:2.5.1(CUDA 12.1)
* torchvision:0.20.1(CUDA 12.1)

## Data setting
Please download [Dataset](https://www.kaggle.com/competitions/taica-cvpdl-2025-hw-1/data) and organize them as following:
```
src/
└── datasets/
    └── images/
        ├── train/
        │   ├── img/
        │   └── gt.txt
        └── test/
            └── img/   
```

## Installation
Use Python >= 3.10.
```bash
cd hw1_M11315025/code_M11315025
conda create -n CVPDL python=3.10.0
conda activate CVPDL
```
Installation of required packages:
```bash
pip install -r requirements.txt
```

## Training
1. Perform data preprocessing
```bash
cd src
python preprocessing.py
```
2. Data segmentation
```bash
python split.py
```
3. Training and Predict
```bash
python model.py
```