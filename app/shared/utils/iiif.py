import glob
import os
import time
import requests

from pathlib import Path
from PIL import Image, UnidentifiedImageError
from urllib.parse import urlparse
from typing import Optional, Tuple, List

# TODO change to not only use IMG_PATH from regions
from ...regions.const import IMG_PATH
from .fileutils import check_dir, sanitize_str, sanitize_url
from .logging import console


def is_iiif_manifest(json_content):
    try:
        return "@context" in json_content
    except (requests.RequestException, ValueError, KeyError):
        return False


def get_json(url):
    try:
        response = requests.get(url)
        if response.ok:
            return response.json()
        else:
            response.raise_for_status()
    except requests.exceptions.RequestException:
        console(f"Error getting JSON for {url}")
        return None


def get_img_rsrc(iiif_img):
    try:
        img_rscr = iiif_img["resource"]
    except KeyError:
        try:
            img_rscr = iiif_img["body"]
        except KeyError:
            return None
    return img_rscr


def get_canvas_img(canvas_img, only_img_url=False):
    img_url = get_id(canvas_img["resource"]["service"])
    if only_img_url:
        return img_url
    return get_img_id(canvas_img["resource"]), img_url


def get_item_img(item_img, only_img_url=False):
    img_url = get_id(item_img["body"]["service"][0])
    if only_img_url:
        return img_url
    return get_img_id(item_img), img_url


def get_img_id(img):
    img_id = get_id(img)
    if ".jpg" in img_id:
        try:
            return img_id.split("/")[-5]
        except IndexError:
            return None
    return img_id.split("/")[-1]


def get_iiif_resources(manifest, only_img_url=False):
    """
    Get all image resources from a IIIF manifest
    """
    try:
        # Usually images URL are contained in the "canvases" field
        img_list = [canvas["images"] for canvas in manifest["sequences"][0]["canvases"]]
        img_info = [get_img_rsrc(img) for imgs in img_list for img in imgs]
    except KeyError:
        # But sometimes in the "items" field
        try:
            img_list = [
                item
                for items in manifest["items"]
                for item in items["items"][0]["items"]
            ]
            img_info = [get_img_rsrc(img) for img in img_list]
        except KeyError:
            console(f"Unable to retrieve resources from manifest {manifest}")
            return []

    return img_info


def get_reduced_size(size, min_size=1500):
    size = int(size)
    if size < min_size:
        return ""
    if size > min_size * 2:
        return str(int(size / 2))
    return str(min_size)


def get_id(dic):
    """
    Get the id of a IIIF resource
    """
    if isinstance(dic, list):
        dic = dic[0]

    if isinstance(dic, dict):
        try:
            return dic["@id"]
        except KeyError:
            try:
                return dic["id"]
            except KeyError:
                console("No id provided")

    if isinstance(dic, str):
        return dic

    return None


class IIIFDownloader:
    """Download all image resources from a list of manifest urls."""

    def __init__(self, manifest_url, target_path=None, img_dir=None, width=None, height=None, sleep=0.25, max_dim=None):
        """
        Args:

        manifest_url (str): URL of the IIIF manifest
        target_path (str): Path where to save the images (if None, use img_dir/manifest_id)
        img_dir (str|Path): Path where to save the images (ignored if target_path is set)
        width (int): Width of the images to download (optional)
        height (int): Height of the images to download (optional)
        sleep (float): Time to sleep between each download (default: 0.25s)
        max_dim (int): Maximal height of the images to download (optional)
        """
        self.manifest_url = manifest_url
        self.manifest_id = ""  # Prefix to be used for img filenames

        if target_path is None:
            if img_dir is None:
                img_dir = IMG_PATH
            target_path = Path(img_dir) / self.get_dir_name()
        self.manifest_dir_path = target_path

        self.size = self.get_formatted_size(width, height)
        self.sleep = sleep
        self.max_dim = max_dim  # Maximal height in px

    def run(self) -> List[Tuple[str, str]]:
        manifest = self.check_url()
        return self.download_manifest(manifest)

    def get_dir_name(self):
        return sanitize_str(self.manifest_url).replace("manifest", "").replace("json", "")

    def check_url(self):
        try:
            manifest = get_json(self.manifest_url)
            if not manifest:
                raise ValueError("Failed to load json content")
            if not is_iiif_manifest(manifest):
                raise ValueError("URL content is not a valid IIIF manifest")
        except Exception as e:
            raise ValueError(f"Failed to load manifest: {e}")
        return manifest

    def download_manifest(self, manifest) -> List[Tuple[str, str]]:
        """
        Download all images from a IIIF manifest
        """
        self.manifest_id = self.manifest_id + self.get_manifest_id(manifest)
        all_img_mapping = []
        if manifest is not None:
            console(f"Processing {self.manifest_url}...")
            if not check_dir(self.manifest_dir_path):
                i = 1
                for rsrc in get_iiif_resources(manifest):
                    is_downloaded, img_name, img_url = self.save_iiif_img(rsrc, i)
                    i += 1
                    if img_name is not None:
                        all_img_mapping.append((img_name, img_url))
                    if is_downloaded:
                        # Gallica is not accepting more than 5 downloads of >1000px per min
                        time.sleep(12 if "gallica" in self.manifest_url else 0.25)
                        time.sleep(self.sleep)
        return all_img_mapping

    def get_formatted_size(self, width="", height=""):
        if not hasattr(self, "max_dim"):
            self.max_dim = None

        if not width and not height:
            if self.max_dim is not None:
                return f",{self.max_dim}"
            return "full"

        if self.max_dim is not None and int(width) > self.max_dim:
            width = f"{self.max_dim}"
        if self.max_dim is not None and int(height) > self.max_dim:
            height = f"{self.max_dim}"

        return f"{width or ''},{height or ''}"

    def save_iiif_img(self, img_rscr, i, size="full", re_download=False) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Save an image from a IIIF resource, at a given size
        """
        if ".jpg" in get_img_id(img_rscr):
            img_name = get_img_id(img_rscr)
        else:
            img_name = f"{self.manifest_id}_{i:04d}.jpg"

        img_url = get_id(img_rscr["service"])
        iiif_url = sanitize_url(f"{img_url}/full/{size}/0/default.jpg")

        if glob.glob(os.path.join(self.manifest_dir_path, f"*_{i:04d}.jpg")) and not re_download:
            # if the img is already downloaded, don't download it again
            return False, img_name, iiif_url

        with requests.get(iiif_url, stream=True) as response:
            response.raw.decode_content = True
            try:
                img = Image.open(response.raw)
                # img.verify()
            except (UnidentifiedImageError, SyntaxError):
                if size == "full":
                    size = get_reduced_size(img_rscr["width"])
                    return self.save_iiif_img(img_rscr, i, self.get_formatted_size(size))
                else:
                    console(f"{iiif_url} is not a valid img file")
                    return False, None, None
            except (IOError, OSError):
                if size == "full":
                    size = get_reduced_size(img_rscr["width"])
                    return self.save_iiif_img(img_rscr, i, self.get_formatted_size(size))
                else:
                    console(f"{iiif_url} is a truncated or corrupted image")
                    return False, None, None

            self.save_img(img, img_name, f"Failed to save {iiif_url}")
        return (True, img_name, iiif_url)

    def save_img(self, img: Image, img_filename, error_msg="Failed to save img"):
        try:
            img.save(self.manifest_dir_path / img_filename)
            return True
        except Exception:
            console(f"{error_msg}")
        return False

    def get_manifest_id(self, manifest):
        manifest_id = get_id(manifest)
        if manifest_id is None:
            return self.get_dir_name()
        if "manifest" in manifest_id:
            try:
                manifest_id = Path(urlparse(get_id(manifest)).path).parent.name
                if "manifest" in manifest_id:
                    return self.get_dir_name()
                return sanitize_str(manifest_id)
            except Exception:
                return self.get_dir_name()
        return sanitize_str(manifest_id.split("/")[-1])
