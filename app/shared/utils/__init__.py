"""
This module contains all the shared functions and classes used by the other modules.
"""

import hashlib


def hash_str(string: str) -> str:
    """
    Hashes a string using the SHA-256 algorithm.
    """
    hash_object = hashlib.sha256()
    hash_object.update(string.encode("utf-8"))
    return hash_object.hexdigest()


def get_device() -> str:
    """
    Returns the device on which to run the code (GPU if available, else CPU)
    """
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"
