import os
import sys
from itertools import combinations_with_replacement
from tqdm import tqdm

import requests
import urllib.request
import json
import shutil

from pathlib import Path
from PIL import Image

# from .similarity import log_failed_img
from ..const import IMG_PATH, MODEL_PATH, LIB_PATH
from .const import MAX_SIZE
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


def save_img(
    img: Image,
    img_filename,
    img_path=IMG_PATH,
    max_dim=MAX_SIZE,
    img_format="JPEG",
):
    try:
        if img.mode != "RGB":
            img = img.convert("RGB")

        if img.width > max_dim or img.height > max_dim:
            img.thumbnail(
                (max_dim, max_dim), Image.ANTIALIAS
            )  # Image.Resampling.LANCZOS

        # TODO use this way of resizing images and remove resize in segswap code
        # tr_ = transforms.Resize((224, 224))
        # img = cv2.imread(query_img)
        # img = torch.from_numpy(img).permute(2, 0, 1)
        # tr_img = tr_(img).permute(1, 2, 0).numpy()
        # cv2.imwrite(query_img, tr_img)

        img.save(f"{img_path}/{img_filename}.jpg", format=img_format)
        return img
    except Exception as e:
        console(f"Failed to save {img_filename} as JPEG", e=e)
        return False


def get_json(url):
    with urllib.request.urlopen(url) as url:
        return json.loads(url.read().decode())


# def hash_pair(pair: tuple):
#     if isinstance(pair, tuple) and len(pair) == 2 and all(isinstance(s, str) for s in pair):
#         return hash_str(''.join(sorted(pair)))
#     raise ValueError("Not a correct pair of document id")


def doc_pairs(doc_ids: list):
    if isinstance(doc_ids, list) and len(doc_ids) > 0:
        return list(combinations_with_replacement(doc_ids, 2))
    raise ValueError("Input must be a non-empty list of ids.")


def download_img(img_url, doc_id, img_name):
    doc_dir = f"{IMG_PATH}/{doc_id}"
    try:
        with requests.get(img_url, stream=True) as response:
            response.raw.decode_content = True
            img = Image.open(response.raw)
            save_img(img, img_name, doc_dir)

    except requests.exceptions.RequestException as e:
        shutil.copyfile(
            f"{LIB_PATH}/media/placeholder.jpg",
            f"{doc_dir}/{img_name}",
        )
        # log_failed_img(img_name, img_url)
        console(f"[download_img] {img_url} is not a valid img file", e=e)
    except Exception as e:
        shutil.copyfile(
            f"{LIB_PATH}/media/placeholder.jpg",
            f"{doc_dir}/{img_name}",
        )
        # log_failed_img(img_name, img_url)
        console(f"[download_img] {img_url} image was not downloaded", e=e)


def download_images(url, doc_id):
    """
    e.g.
    url = https://eida.obspm.fr/eida/wit1_man191_anno188/list/
    images = {
        "img_name": "https://domain-name.com/image_name.jpg",
        "img_name": "https://other-domain.com/image_name.jpg",
        "img_name": "https://iiif-server.com/.../coordinates/size/rotation/default.jpg",
        "img_name": "..."
    }
    """

    images = get_json(url)
    if len(images.items()) == 0:
        console(f"{url} does not contain any images.", color="yellow")
        return []

    # i = 1
    paths = []
    for (
        img_name,
        img_url,
    ) in images.items():  # tqdm(images.items(), desc="Downloading Images"):
        # img_name = f"{i:0{z}}.jpg"
        # i += 1
        download_img(img_url, doc_id, img_name)
        paths.append(f"{IMG_PATH}/{doc_id}/{img_name}")

    return paths


def get_img_paths(img_dir):
    images = []
    for file_ in os.listdir(img_dir):
        if file_.endswith((".jpg", ".png", ".jpeg")):
            images.append(os.path.join(img_dir, file_))
        else:
            sys.stderr.write(
                f"Image format is not compatible in {file_}. Skipping this file.\n"
            )
    return sorted(images)


def get_imgs_in_dirs(img_dirs):
    images = []
    for img_dir in img_dirs:
        images.extend(get_img_paths(img_dir))
    return images


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
