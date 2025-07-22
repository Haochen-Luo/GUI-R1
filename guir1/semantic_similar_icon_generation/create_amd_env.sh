#!/bin/bash
# Create conda env
conda create -y -n amd-diffusion python=3.9 pillow
conda activate amd-diffusion

# Install PyTorch ROCm build
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0

# Install Huggingface + other deps
pip install diffusers transformers accelerate datasets huggingface_hub

# (Optional) Test torch
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"