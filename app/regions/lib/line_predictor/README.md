# Line Predictor

## Setup

In order to use line predictor for extraction, you first need to compile CUDA operators

1. Set CUDA_HOME (usually located either in `/usr/local/cuda` or `/usr/lib/cuda`):
    ```bash
    echo $CUDA_HOME  # if already defined, go to next step

    # find CUDA version with (pay attention to version mismatches)
    nvcc --version
    nvidia-smi

    # set CUDA_HOME
    export CUDA_HOME=<path/to/cuda>
    ```
2. Activate python environment in the api root folder
    ```bash
    source venv/bin/activate
    ```
3. Inside `line_predictor/` folder, run
    ```bash
    # Compile CUDA operators
    python ./models/dino/ops/setup.py build install
    # Unit test => could output an outofmemory error
    python ./models/dino/ops/test.py
    ```

## Copyright

> ### Line Predictor
> Copyright (c) 2024 Raphaël Baena (Imagine team - LIGM)
> Licensed under the Apache License, Version 2.0
> Copied from LinePredictor (https://github.com/raphael-baena/LinePredictor)
