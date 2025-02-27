# Final image (change image based on the version showed with $ nvidia-smi)
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV http_proxy=http://proxy.enpc.fr:3128
ENV https_proxy=http://proxy.enpc.fr:3128
ENV HTTP_PROXY=http://proxy.enpc.fr:3128
ENV HTTPS_PROXY=http://proxy.enpc.fr:3128

ENV USER=demoapi
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
    git \
    poppler-utils

WORKDIR /home/${USER}

## Set nvidia
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

## Copy requirements and install them
# (Before copying the rest so that this part is not rerun unless requirements change)
COPY --chown=${USER} ./requirements.txt ./requirements.txt
COPY --chown=${USER} ./requirements-prod.txt ./requirements-prod.txt
RUN python3.10 -m venv venv && \
    source venv/bin/activate && \
    pip install --upgrade pip && \
    pip install -r requirements-prod.txt

# Copy the entire project code
COPY --chown=${USER} ./ ./api/

# FOR VECTORIZATION MODULE
# Build and install CUDA operators for DETR (not working)
#RUN /home/${USER}/venv/bin/python /home/${USER}/api/app/vectorization/lib/src/models/dino/ops/setup.py build install
#RUN /home/${USER}/venv/bin/python /home/${USER}/api/app/vectorization/lib/src/models/dino/ops/test.py
#RUN /home/${USER}/venv/bin/pip install -e /home/${USER}/api/app/vectorization/lib/synthetic/

WORKDIR /home/${USER}

# Copy additional configurations
COPY --chown=${USER} ./.env.prod ./api/.env
COPY docker-confs/nginx.conf /etc/nginx/conf.d/demoapi.conf

# Expose the application port
EXPOSE 8001

# Set matplotlib tmp dir
ENV MPLCONFIGDIR=/home/${USER}/.config/matplotlib
RUN mkdir -p /home/${USER}/.config/matplotlib
RUN chown -R ${USER} /home/${USER}/.config/matplotlib

# Create necessary folders
RUN mkdir -p var/dramatiq/

# Add these lines after installing ffmpeg
RUN mkdir -p /usr/lib/x86_64-linux-gnu/gstreamer-1.0/gstreamer-1.0/
RUN chmod 755 /usr/lib/x86_64-linux-gnu/gstreamer-1.0/gstreamer-1.0/

# Add GStreamer environment variables
ENV GST_PLUGIN_SYSTEM_PATH=/usr/lib/x86_64-linux-gnu/gstreamer-1.0
ENV GST_PLUGIN_SCANNER=/usr/lib/x86_64-linux-gnu/gstreamer1.0/gstreamer-1.0/gst-plugin-scanner
ENV GST_REGISTRY=/home/${USER}/.cache/gstreamer-1.0/registry.x86_64.bin

# Create and set permissions for GStreamer cache directory
RUN mkdir -p /home/${USER}/.cache/gstreamer-1.0
RUN chown -R ${USER}:${USER} /home/${USER}/.cache

# Run command at each container launch
CMD export LC_ALL=C.UTF-8 && export LANG=C.UTF-8 && \
    source venv/bin/activate && \
    supervisord -c api/docker-confs/supervisord.conf
