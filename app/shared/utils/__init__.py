"""
This module contains all the shared functions and classes used by the other modules.
"""

import hashlib
import torch
import gc


def hash_str(string: str | bytes) -> str:
    """
    Hashes a string using the SHA-256 algorithm.
    """
    if isinstance(string, str):
        string = string.encode("utf-8")
    hash_object = hashlib.sha256()
    hash_object.update(string)
    return hash_object.hexdigest()


def get_device() -> str:
    """
    Returns the device on which to run the code (GPU if available, else CPU)
    """
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def clear_cuda():
    """
    Clears the CUDA cache and garbage collects GPU memory.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
