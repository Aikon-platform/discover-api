#!/bin/bash

###############################
# VALUES TO BE USED BY DOCKER #
#    modify those values to   #
#      match your system      #
###############################

# Machine path where docker will store its /data/ folder (API_DATA_FOLDER)
DATA_FOLDER=/media/demoapi/
# GPU device number to be used by docker
DEVICE_NB=2
# User ID to be used by docker
DEMO_UID=1000
# Path to CUDA installation (/usr/local/cuda-<version> => find your version with nvidia-smi)
CUDA_HOME=/usr/local/cuda
# Host where the API will be accessible, (put 127.0.0.1 for spiped configuration)
CONTAINER_HOST="0.0.0.0"
# Name of the container
CONTAINER_NAME="demoapi"

rebuild_image() {
    # Add error checking for the build process
    if ! docker build --rm -t "$CONTAINER_NAME" . -f Dockerfile --build-arg USERID=$DEMO_UID; then
        echo "Docker build failed"
        exit 1
    fi
    cd ../
}

# if container exists and stop it
if docker ps -a --format '{{.Names}}' | grep -Eq "$CONTAINER_NAME"; then
    docker stop "$CONTAINER_NAME"
    docker rm "$CONTAINER_NAME"
fi

if [ "$1" = "rebuild" ]; then
    rebuild_image
fi

if [ "$1" = "pull" ]; then
    git pull
    git submodule update
    rebuild_image
fi

# Only run the container if it exists
if docker image inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
    # Run Docker container
    docker run -d --gpus "$DEVICE_NB" --name "$CONTAINER_NAME" \
       -v "$DATA_FOLDER":/data/ -v "$CUDA_HOME":/cuda/ -p "$CONTAINER_HOST":8001:8001 \
       --restart unless-stopped --ipc=host "$CONTAINER_NAME"
else
    echo "Image $CONTAINER_NAME does not exist. Build failed or not yet built."
    exit 1
fi