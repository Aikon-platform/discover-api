#!/bin/bash

[ ! -f .env.dev ] || export $(grep -v '^#' .env.dev | xargs)

flask_pid=""
dramatiq_pid=""

cleanup() {
    echo "Shutting down processes..."
    [ -n "$flask_pid" ] && kill "$flask_pid"
    [ -n "$dramatiq_pid" ] && kill "$dramatiq_pid"
    wait
    echo "All processes terminated."
    exit 0
}

trap cleanup SIGINT SIGTERM

export CUDA_VISIBLE_DEVICES=$DEVICE_NB
venv/bin/flask --app app.main run --debug -p "$API_DEV_PORT" &
flask_pid=$!

venv/bin/dramatiq app.main -t 1 -p 1 &
dramatiq_pid=$!

wait
