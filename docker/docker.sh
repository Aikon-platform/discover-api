#!/bin/bash

# HOW TO USE
# Inside the docker/ directory, run:
# bash docker.sh <start|stop|restart|update|build>

set -e

DOCKER_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# initialize the .env files and data folder permissions on first initialization
bash "$DOCKER_DIR"/init.sh

# Load container variables from .env file
. "$DOCKER_DIR"/.env

rebuild_image() {
    # Add error checking for the build process
    docker build --rm -t "$CONTAINER_NAME" -f Dockerfile .. \
        --build-arg USERID=$DEMO_UID \
        --build-arg HTTP_PROXY=${HTTP_PROXY} \
        --build-arg HTTPS_PROXY=${HTTPS_PROXY} \
        --build-arg NO_PROXY=${NO_PROXY} \
        --build-arg HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN} || { echo "Docker build failed"; exit 1; }
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
