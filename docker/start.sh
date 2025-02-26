#!/bin/bash

set -e

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source "$ROOT_DIR"/utils.sh
source "$ROOT_DIR"/api/.env

# Create necessary directories at startup
mkdir -p "$ROOT_DIR"/var/dramatiq/
mkdir -p "$ROOT_DIR"/.config/matplotlib
chown -R $USER "$ROOT_DIR"/.config/matplotlib

source "$ROOT_DIR"/venv/bin/activate

if [[ "$INSTALLED_APPS" == *"vectorization"* ]]; then
    color_echo blue "Building operators for vectorization module..."
    cd "$ROOT_DIR"/api/app/vectorization/lib/
    python src/models/dino/ops/setup.py build install
    # python src/models/dino/ops/test.py
    # pip install -e synthetic/
fi

if [[ "$INSTALLED_APPS" == *"regions"* ]]; then
    color_echo blue "Building operators for regions module..."
    cd "$ROOT_DIR"/api/app/regions/lib/line_predictor/dino/ops/
    python setup.py build install
fi

# Run command at each container launch
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
supervisord -c "$ROOT_DIR"/supervisord.conf
