# Paper Submission 3071


## Overview

This repository provides the validation code for the **single-modal data co-training (IPT-based)** experiments and the **multi-modal data co-training (CNN-classification)** experiments mentioned in the paper, along with the corresponding datasets. All deep learning code is built using PyTorch. The datasets and model weights are available at: https://drive.google.com/drive/folders/1zqDvtqSz1yj1Kz_tgqGK9aTbnNl0x1Wm?usp=drive_link.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Usage](#usage)
  - [Model Weights](#model-weights)
  - [Datasets](#datasets)
  - [Testing](#testing)

## Prerequisites

Ensure you have the following dependencies installed:

For **single-modal data co-trianing**: 
- Python 3.x
- PyTorch 1.10.1
- Additional libraries are listed in the `requirements_1.txt` file and can be installed with:

  ```bash
  pip install -r requirements_1.txt
For **multi-modal data co-trianing**: 
- Python 3.x
- PyTorch 1.9.1
- Additional libraries are listed in the `requirements_2.txt` file and can be installed with:

  ```bash
  pip install -r requirements_2.txt
## Usage

### Model Weights
We provide the model weights for **single-modal data co-training (IPT-based)**: `finetune_x2_best_epoch.pt`, `finetune_x3_best_epoch.pt`, `finetune_x4_best_epoch.pt`, `epoch_48.pt`.
We also provide the model weights for **multi-modal data co-training (CNN-classification)**, including the full-precision ResNet34 weights `resnet34.pth`, the ResNet34 weights quantized using LSQ+ `lsq+_resnet34.pth`, and the ResNet34 weights quantized using TSQ-MTC `TSQ_MTC_resnet34.pth`.

### Datasets
For **single-modal data co-trianing**: 
- `benchmark.zip`, unzip to root directory of TSQ_MTC_IPT

For **multi-modal data co-trianing**:
- We provide the dataset used for testing the model in SAR_RGB, `SAR_RGB_dataset.zip`, which contains various RGB images and SAR images, detailed introduced in the paper. 

### Testing
For **single-modal data co-trianing**: 
- To evaluate the model of IPT，please use the following command:
    ```
    sh eval_x2.sh # for x2 SR
    sh eval_x3.sh # for x3 SR
    sh eval_x4.sh # for x4 SR
    sh eval_derain.sh # for deraining
    sh eval_denoise30.sh # for denoising with sigma=30
    sh eval_denoise50.sh # for denoising with sigma=30
    ```

For **multi-modal data co-trianing**:
- To test the resnet34-based co-training model using LSQ+, please modify test.sh and set `--data_dir` to /path/to/SAR_RGB, `--ckpt_path` to /path/to/lsq+_resnet34.pth, and use the following command:
    ```
    bash test.sh
    ```
- To test the resnet34-based co-training model using TSQ-MTC, please modify test.sh and set --data_dir to /path/to/SAR_RGB, --ckpt_path to /path/to/TSQ_MTC_resnet34.pth, and use the following command:
    ```
    bash test.sh
    ```
