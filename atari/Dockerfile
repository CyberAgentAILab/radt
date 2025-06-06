FROM nvidia/cuda:12.6.1-cudnn-devel-ubuntu22.04
WORKDIR /radt_atari

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get -y update --fix-missing && apt-get upgrade -y

RUN apt-get -y install --no-install-recommends python3.10-dev python3-pip
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

RUN apt-get update --fix-missing && apt-get upgrade -y && \
    apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion nano libz-dev \
    libosmesa6-dev libgl1-mesa-glx libglfw3 build-essential  && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
ENV CMAKE_POLICY_VERSION_MINIMUM 3.5
RUN pip install atari-py==0.2.9
RUN pip install git+https://github.com/google/dopamine.git@bec5f4e108b0572e58fc1af73136e978237c8463
RUN pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126

COPY . /radt_atari

RUN python3 -m atari_py.import_roms /radt_atari/roms/

RUN chmod -R 777 /radt_atari/scripts
