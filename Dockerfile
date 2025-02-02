FROM nvidia/cuda:12.5.1-cudnn-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    git-lfs \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth=1 --branch main https://github.com/runarheggset/MuseTalk /app && rm -rf /app/.git
RUN git clone --depth=1 --branch main https://huggingface.co/TMElyralab/MuseTalk /app/models && rm -rf /app/models/.git

WORKDIR /app/models/dwpose
RUN wget https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.pth

WORKDIR /app/models/face-parse-bisent
RUN wget https://download.pytorch.org/models/resnet18-5c106cde.pth
RUN wget https://github.com/zllrunning/face-makeup.PyTorch/raw/master/cp/79999_iter.pth

WORKDIR /app/models/sd-vae-ft-mse
RUN wget https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.safetensors
RUN wget https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/config.json

WORKDIR /app/models/whisper
RUN wget https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt

WORKDIR /app

RUN python3 -m venv venv
ENV PATH="/app/venv/bin:$PATH"

RUN pip install -r requirements.txt
RUN pip install openmim psutil ninja
RUN mim install "mmengine==0.10.6" "mmcv==2.1.0" "mmdet==3.2.0" "mmpose==1.3.2"

CMD ["python", "server.py"]
