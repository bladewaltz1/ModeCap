# Learning Distinct and Representative Modes for Image Captioning (Neurips 2022)

This repo provides the implemetation of the paper [Learning Distinct and Representative Modes for Image Captioning](https://arxiv.org/abs/2209.08231).

## Install

```bash
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116
pip install transformers yacs scipy
```

## Data
Follow the instructions in [VLP](https://github.com/LuoweiZhou/VLP#-data-preparation).

## Run
```bash
python -m modecap.train data_dir PATH_TO_DATA
python -m modecap.inference data_dir PATH_TO_DATA model_path PATH_TO_MODEL
```
