import hashlib


def hash_str(string):
    hash_object = hashlib.sha256()
    hash_object.update(string.encode("utf-8"))
    return hash_object.hexdigest()


def get_device():
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"
