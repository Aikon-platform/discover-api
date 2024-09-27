import os
import sys

import requests
import urllib.request
import json
import shutil

from PIL import Image

from ..const import UTILS_DIR
from .logging import console


MAX_SIZE = 244


def get_json(url):
    with urllib.request.urlopen(url) as url:
        return json.loads(url.read().decode())


def save_img(
    img: Image,
    img_filename,
    img_path,
    max_dim=MAX_SIZE,
    img_format="JPEG",
):
    try:
        if img.mode != "RGB":
            img = img.convert("RGB")

        if img.width > max_dim or img.height > max_dim:
            img.thumbnail(
                (max_dim, max_dim), Image.Resampling.LANCZOS
            )  # Image.ANTIALIAS

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


def download_img(img_url, doc_id, img_name, img_path, max_dim=MAX_SIZE):
    doc_dir = f"{img_path}/{doc_id}"
    try:
        with requests.get(img_url, stream=True) as response:
            response.raw.decode_content = True
            img = Image.open(response.raw)
            save_img(img, img_name, doc_dir, max_dim)

    except requests.exceptions.RequestException as e:
        shutil.copyfile(
            f"{UTILS_DIR}/img/placeholder.jpg",
            f"{doc_dir}/{img_name}",
        )
        # log_failed_img(f"{doc_dir}/{img_name}", img_url)
        console(f"[download_img] {img_url} is not a valid img file", e=e)
    except Exception as e:
        shutil.copyfile(
            f"{UTILS_DIR}/img/placeholder.jpg",
            f"{doc_dir}/{img_name}",
        )
        # log_failed_img(f"{doc_dir}/{img_name}", img_url)
        console(f"[download_img] {img_url} image was not downloaded", e=e)


def download_images(url, doc_id, img_path, max_dim=MAX_SIZE):
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
        download_img(img_url, doc_id, img_name, img_path, max_dim)
        paths.append(f"{img_path}/{doc_id}/{img_name}")

    return paths


def get_img_paths(img_dir, img_ext=(".jpg", ".png", ".jpeg")):
    images = []
    for file_ in os.listdir(img_dir):
        if file_.endswith(img_ext):
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
