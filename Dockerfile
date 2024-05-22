FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git

# Retry function for installing packages
RUN set -eux; \
    for i in $(seq 1 5); do \
        pip3 install --default-timeout=1000 torch==2.3.0+cu118 torchvision==0.14.0+cu118 torchaudio==0.13.0+cu118 -f https://download.pytorch.org/whl/cu118/torch_stable.html && break || sleep 10; \
    done

# Install flash_attn dependencies
RUN set -eux; \
    for i in $(seq 1 5); do \
        pip3 install flash_attn --default-timeout=1000 && break || sleep 10; \
    done

# Clone and install flash_attn
RUN git clone https://github.com/Dao-AILab/flash-attention.git
WORKDIR /flash-attention
RUN python3 setup.py install

# Set the working directory
WORKDIR /workspace
