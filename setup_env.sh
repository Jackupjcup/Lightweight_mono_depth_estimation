#!/bin/bash

# conda create -n fast_mono_depth_lrd python=3.11 -y
# conda activate fast_mono_depth_lrd
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# pip install timm einops omegaconf addict safetensors huggingface_hub
# pip install opencv-python tensorboard tqdm pillow numpy scipy matplotlib lmdb
# pip install xformers


set -e

ENV_NAME="fast_mono_depth_lrd"

conda create -n ${ENV_NAME} python=3.11 -y
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install timm einops omegaconf addict safetensors huggingface_hub
pip install opencv-python tensorboard tqdm pillow numpy scipy matplotlib lmdb
pip install xformers
