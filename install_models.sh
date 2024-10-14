#!/bin/bash

# Create directories if they don't exist
mkdir -p models/vae models/clip models/unet models/checkpoints models/upscale_models

# Download VAE
cd models/vae
wget https://huggingface.co/oguzm/flux.dev1/resolve/main/FLUX.1-dev/ae.safetensors -O ae.safetensors

# Download CLIP models
cd ../clip
wget https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors -O clip_l.safetensors
wget https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors -O t5xxl_fp16.safetensors

# Download UNET
cd ../unet
wget https://huggingface.co/oguzm/flux.dev1/resolve/main/FLUX.1-dev/flux1-dev.safetensors -O flux1-dev.safetensors

# Download checkpoint
cd ../checkpoints
wget https://civitai.com/api/download/models/134065 -O epicrealism-v5.safetensors

cd ../upscale_models
wget https://civitai.com/api/download/models/125843 -O 4x-UltraSharp.pth

echo "All downloads completed!"