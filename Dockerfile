# Final image (change image based on the version showed with $ nvidia-smi)
FROM nvidia/cuda:11.6.1-cudnn8-devel-ubuntu20.04
#FROM nvidia/cuda:12.4.0-cudnn8-devel-ubuntu20.04

ENV USER=demowebsite
ARG USERID

RUN useradd -u ${USERID} -m -d /home/${USER} ${USER}

## Silence error messages
ENV TERM=linux

## Bash instead of shell
SHELL ["/bin/bash", "-c"]

## Install utils
ENV TZ=Europe/Paris
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt install -y python3.10

RUN apt-get update && apt-get install -y \
    unzip \
    wget \
    zip \
    supervisor \
    ffmpeg \
    libsm6 \
    libxext6 \
    build-essential \
    python3.10-dev \
    python3.10-venv \
    redis-server \
    nginx \
    git

WORKDIR /home/${USER}

## Set nvidia
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

## Copy requirements and install them
# (Before copying the rest so that this part is not rerun unless requirements change)
COPY --chown=${USER} ./requirements.txt ./requirements.txt
COPY --chown=${USER} ./requirements-prod.txt ./requirements-prod.txt
RUN python3.10 -m venv venv
RUN source venv/bin/activate && pip install --upgrade pip && pip install -r requirements-prod.txt

# Install components
COPY --chown=${USER} ./ ./api/
COPY --chown=${USER} ./.env.prod ./api/.env

COPY docker-confs/nginx.conf /etc/nginx/conf.d/demowebsite.conf

EXPOSE 8001

RUN mkdir -p var/dramatiq/

ENV CUDA_HOME=/cuda

## Run command at each container launch
CMD export LC_ALL=C.UTF-8 && export LANG=C.UTF-8 && \
    source venv/bin/activate && \
    supervisord -c api/docker-confs/supervisord.conf
