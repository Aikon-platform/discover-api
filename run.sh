#! /bin/bash
[ ! -f .env.dev ] || export $(grep -v '^#' .env.dev | xargs)

(trap 'kill 0' SIGINT;
    (export CUDA_VISIBLE_DEVICES=$DEVICE_NB && venv/bin/flask --app app.main run --debug -p $DEV_PORT) &
    (venv/bin/dramatiq app.main -t 1 -p 1) &
);