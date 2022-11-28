FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

WORKDIR /app
ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt update \
    && apt install -y wget curl git make build-essential \
        libgl1-mesa-dev libglib2.0-0

RUN apt install -y python3-distutils python3.9 python3.9-dev \
    && wget -O ~/get-pip.py https://bootstrap.pypa.io/get-pip.py \
    && python3.9 ~/get-pip.py \
    && rm ~/get-pip.py \
    && ln -sf /usr/bin/python3.9 /usr/local/bin/python3 \
    && ln -sf /usr/bin/python3.9 /usr/local/bin/python \
    && python -m pip install --upgrade pip setuptools --no-cache-dir \
    && python -m pip install wheel --no-cache-dir

COPY requirements.txt ./

RUN pip install -r requirements.txt \
    && pip uninstall -y torch torchvision \
    && pip install torch==1.10.2 torchvision==0.11.3 --extra-index-url https://download.pytorch.org/whl/cu113
