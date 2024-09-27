import os
import sys
from itertools import combinations_with_replacement

import requests

from pathlib import Path

from ..const import IMG_PATH, MODEL_PATH
from ...shared.utils.fileutils import create_dir
from ...shared.utils.logging import console

model_urls = {
    "moco_v2_800ep_pretrain": "https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_200ep/moco_v2_200ep_pretrain.pth.tar",
    "dino_deitsmall16_pretrain": "https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth",
    "dino_vitbase8_pretrain": "https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth",
    "hard_mining_neg5": "https://github.com/XiSHEN0220/SegSwap/raw/main/model/hard_mining_neg5.pth",
}


def download_models(model_name):
    os.makedirs(f"{MODEL_PATH}/", exist_ok=True)

    if model_name not in model_urls:
        raise ValueError("Invalid network or dataset for feature extraction.")

    response = requests.get(model_urls[model_name])
    if response.status_code == 200:
        with open(f"{MODEL_PATH}/{model_name}.pth", "wb") as file:
            file.write(response.content)
        return
    console(f"Failed to download the file. Status code: {response.status_code}", "red")


def get_model_path(model_name):
    if model_name not in model_urls:
        sys.stderr.write("Invalid network or dataset for feature extraction.")
        exit()

    if not os.path.exists(f"{MODEL_PATH}/{model_name}.pth"):
        download_models(model_name)

    return f"{MODEL_PATH}/{model_name}.pth"


def doc_pairs(doc_ids: list):
    if isinstance(doc_ids, list) and len(doc_ids) > 0:
        return list(combinations_with_replacement(doc_ids, 2))
    raise ValueError("Input must be a non-empty list of ids.")


def get_doc_dirs(doc_pair):
    return [
        IMG_PATH / doc
        for doc in (doc_pair if doc_pair[0] != doc_pair[1] else [doc_pair[0]])
    ]


def is_downloaded(doc_id):
    path = Path(f"{IMG_PATH}/{doc_id}/")
    if not os.path.exists(path):
        create_dir(path)
        return False
    if len(os.listdir(path)) == 0:
        return False
    return True


def best_matches(segswap_pairs, q_img, doc_pair):
    """
    segswap_pairs = [[score, img_doc1.jpg, img_doc2.jpg]
                     [score, img_doc1.jpg, img_doc2.jpg]
                     ...]
    q_img = "path/to/doc_hash/img_name.jpg"
    doc_pair = (doc1_hash, doc2_hash)
    """
    query_hash = os.path.dirname(q_img).split("/")[-1]
    query_doc = 1 if query_hash == doc_pair[0] else 2
    sim_doc = 2 if query_doc == 1 else 1
    sim_hash = doc_pair[1] if query_hash == doc_pair[0] else doc_pair[0]

    # Get pairs concerning the given query image q_img
    # img_pairs = segswap_pairs[segswap_pairs[:, query_doc] == q_img]
    img_pairs = segswap_pairs[segswap_pairs[:, query_doc] == os.path.basename(q_img)]

    # return sorted([(pair[0], f"{sim_hash}/{pair[sim_doc]}") for pair in img_pairs], key=lambda x: x[0], reverse=True)
    return [(float(pair[0]), f"{sim_hash}/{pair[sim_doc]}") for pair in img_pairs]
