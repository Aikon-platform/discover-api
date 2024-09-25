#!/bin/bash

###############################
# VALUES TO BE USED BY DOCKER #
#    modify those values to   #
#      match your system      #
###############################

# Machine path where docker will store its /data/ folder (API_DATA_FOLDER)
DATA_FOLDER=/media/discoverdemo/
# GPU device number to be used by docker
DEVICE_NB=2
# User ID to be used by docker
DEMO_UID=1000

CONTAINER_NAME="demowebsiteapi"

rebuild_image() {
    docker build --rm -t "$CONTAINER_NAME" . -f Dockerfile --build-arg USERID=$DEMO_UID
    cd ../
}

# if container exists and stop it
if docker ps -a --format '{{.Names}}' | grep -Eq "$CONTAINER_NAME"; then
    docker stop "$CONTAINER_NAME"
fi

if [ "$1" = "rebuild" ]; then
    rebuild_image
fi

if [ "$1" = "pull" ]; then
    git pull
    rebuild_image
fi

docker rm "$CONTAINER_NAME"

# Run Docker container
docker run -d --gpus "$DEVICE_NB" --name "$CONTAINER_NAME" \
   -v "$DATA_FOLDER":/data/ -v "$CUDA_HOME":/cuda/ -p 127.0.0.1:8001:8001 \
   --restart unless-stopped --ipc=host "$CONTAINER_NAME"
