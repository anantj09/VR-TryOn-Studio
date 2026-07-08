FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Avoid prompt questions during package installations
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies (git, build-essential, OpenCV drivers, python, ffmpeg)
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default python/python3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

WORKDIR /app

# Install PyTorch with CUDA 11.8 support first to ensure proper hardware mapping
RUN pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu118

COPY requirements.txt .

# Install dependencies (use --no-build-isolation to allow compilation of basicsr and detectron2)
RUN pip3 install --no-cache-dir -r requirements.txt --no-build-isolation

# Clone CatVTON repository so it is fully self-contained inside the container
RUN git clone https://github.com/Zheng-Chong/CatVTON.git /app/models/CatVTON

COPY . .

EXPOSE 8000

CMD ["python3", "main.py"]
