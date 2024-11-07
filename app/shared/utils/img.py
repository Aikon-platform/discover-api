"""
Tools to download and save images

Many of the functions in this module are deprecated, better use the Document/Dataset interface
"""

import os
import sys
from pathlib import Path
from typing import List

import requests
import shutil

from PIL import Image

from .fileutils import get_all_files, TPath
from ..const import UTILS_DIR
from .logging import console


MAX_SIZE = 244


def get_json(url: str) -> dict:
    """
    Get a JSON object from a URL
    """
    req = requests.get(url)
    return req.json()


def save_img(
    img: Image,
    img_filename: str,
    img_path: TPath,
    max_dim: int = MAX_SIZE,
    img_format: str = "JPEG",
):
    """
    Save an image to a file
    Resize the image if it is larger than the max_dim
    Convert the image to RGB if it is not already in that mode
    Filename should not include the extension

    :param img: PIL Image object
    :param img_filename: The filename of the image
    :param img_path: The directory to save the image
    :param max_dim: The maximum dimension of the image (default: 244)
    :param img_format: The format to save the image as (default: "JPEG")

    :return: The saved image or False if the image could not be saved
    """
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


def download_image(
    img_url: str, target_dir: TPath, target_filename: str, max_dim: int = MAX_SIZE
) -> None:
    """
    Download an image from a URL and save it to a target file
    If the image is not valid, a placeholder image is saved instead
    Filename should not include the extension

    :param img_url: The URL of the image
    :param target_dir: The directory to save the image
    :param target_filename: The filename of the image
    :param max_dim: The maximum dimension of the image (default: 244)
    """
    try:
        with requests.get(img_url, stream=True) as response:
            response.raw.decode_content = True
            img = Image.open(response.raw)
            save_img(img, target_filename, target_dir, max_dim)

    except requests.exceptions.RequestException as e:
        shutil.copyfile(
            f"{UTILS_DIR}/img/placeholder.jpg",
            f"{target_dir}/{target_filename}",
        )
        # log_failed_img(f"{doc_dir}/{img_name}", img_url)
        console(f"[download_img] {img_url} is not a valid img file", e=e)
    except Exception as e:
        shutil.copyfile(
            f"{UTILS_DIR}/img/placeholder.jpg",
            f"{target_dir}/{target_filename}",
        )
        # log_failed_img(f"{doc_dir}/{img_name}", img_url)
        console(f"[download_img] {img_url} image was not downloaded", e=e)


@DeprecationWarning
def download_img(img_url, doc_id, img_name, img_path, max_dim=MAX_SIZE):
    return download_image(img_url, f"{img_path}/{doc_id}", img_name, max_dim)


@DeprecationWarning
def download_images(url: str, doc_id: str, img_path: TPath, max_dim: int=MAX_SIZE):
    """
    Download images from a URL containing a list of images

    .. code-block:: python

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

@DeprecationWarning
def get_img_paths(img_dir, img_ext=(".jpg", ".png", ".jpeg")) -> list[Path]:
    """
    Get all image paths in a directory
    """
    return get_all_files(img_dir, img_ext)

@DeprecationWarning
def get_imgs_in_dirs(img_dirs) -> list[str]:
    images = []
    for img_dir in img_dirs:
        # TODO check if necessary to transform to strings (used only for similarity dataset)
        images.extend([str(img) for img in get_img_paths(img_dir)])
    return images
