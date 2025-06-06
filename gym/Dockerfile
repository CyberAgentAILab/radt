FROM nvidia/cuda:12.6.1-cudnn-devel-ubuntu22.04
WORKDIR /radt_gym

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get -y update --fix-missing && apt-get upgrade -y

RUN apt-get -y install --no-install-recommends python3.10-dev python3-pip
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

RUN apt-get -y install wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion nano \
    libosmesa6-dev libgl1-mesa-glx libglfw3 build-essential \
    python3-opencv

RUN pip install --upgrade pip

# Install MuJoCo
RUN mkdir -p ~/.mujoco/mujoco210 && \
    wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz && \
    tar --no-same-owner -zxvf mujoco210-linux-x86_64.tar.gz && \
    mv mujoco210 ~/.mujoco && \
    rm mujoco210-linux-x86_64.tar.gz
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/lib/nvidia/lib64
RUN pip install mujoco-py==2.1.2.14 && \
    apt-get reinstall libglvnd0 && \
    apt-get install patchelf

RUN apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install git+https://github.com/Farama-Foundation/d4rl.git@71a9549f2091accff93eeff68f1f3ab2c0e0a288
RUN pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126

COPY . /radt_gym

RUN chmod -R 777 /radt_gym/scripts
