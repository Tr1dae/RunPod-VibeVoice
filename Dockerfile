FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# System deps: git for cloning VibeVoice source; ffmpeg for audio decoding
RUN apt-get update && apt-get install -y \
    ffmpeg libsndfile1 git \
    python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip so pyproject.toml packages install correctly on Ubuntu 22.04
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# CUDA PyTorch must be installed before vibevoice; otherwise pip pulls CPU-only wheels.
# cu121 wheels run on CUDA 12.x hosts (12.2 base image). Reinstall after vibevoice too.
RUN pip3 install --no-cache-dir torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# Install VibeVoice community fork (provides the vibevoice package)
RUN git clone https://github.com/vibevoice-community/VibeVoice.git /vibevoice-src \
    && pip3 install --no-cache-dir /vibevoice-src

RUN pip3 install --no-cache-dir torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# Install additional deps (runpod, huggingface-cli, etc.)
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Bake model weights and tokenizer into the image at build time
RUN huggingface-cli download aoi-ot/VibeVoice-Large \
        --local-dir /app/models/VibeVoice-Large \
    && huggingface-cli download Qwen/Qwen2.5-7B \
        --local-dir /app/models/Qwen2.5-7B

# Point processor at the baked-in tokenizer (avoids HF download at cold start)
RUN python3 -c "\
import json; \
p='/app/models/VibeVoice-Large/preprocessor_config.json'; \
c=json.load(open(p)); \
c['language_model_pretrained_name']='/app/models/Qwen2.5-7B'; \
json.dump(c, open(p,'w'), indent=2)"

ENV MODEL_PATH=/app/models/VibeVoice-Large
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

COPY handler.py .

CMD ["python3", "handler.py"]
