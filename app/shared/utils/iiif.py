"""
Utility functions to handle and download IIIF resources
"""

import glob
import os
import time
import requests

from pathlib import Path
from PIL import Image, UnidentifiedImageError
from urllib.parse import urlparse
from typing import Optional, Tuple, List, Union

# TODO change to not only use IMG_PATH from regions
from ...regions.const import IMG_PATH
from .fileutils import check_dir, sanitize_str, sanitize_url, TPath
from .logging import console
import warnings


def is_iiif_manifest(json_content: dict) -> bool:
    """
    Check if a JSON content is a valid IIIF manifest
    """
    try:
        return "@context" in json_content
    except (requests.RequestException, ValueError, KeyError):
        return False


def get_json(url: str) -> Optional[dict]:
    """
    Get JSON content from a URL
    """
    try:
        response = requests.get(url)
        if response.ok:
            return response.json()
        else:
            response.raise_for_status()
    except requests.exceptions.RequestException:
        console(f"Error getting JSON for {url}")
        return None


def get_img_rsrc(iiif_img: dict) -> Optional[dict]:
    """
    Get the image resource from a IIIF image
    """
    try:
        img_rscr = iiif_img["resource"]
    except KeyError:
        try:
            img_rscr = iiif_img["body"]
        except KeyError:
            return None
    return img_rscr


def get_iiif_resources(manifest: dict) -> List[dict]:
    """
    Get all image resources from a IIIF manifest

    Args:
        manifest (dict): The IIIF manifest

    Returns:
        A list of image resources
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


def get_reduced_size(size: Union[int, str], min_size: int=1500) -> str:
    """
    Adapt the size of an image to a given size

    - If the image is larger than 2*min_size, return the size/2
    - If the image is larger than min_size, return min_size
    - Otherwise return ""

    (Used when images are in error)
    """
    size = int(size)
    if size < min_size:
        return ""
    if size > min_size * 2:
        return str(int(size / 2))
    return str(min_size)


def get_id(dic: dict) -> Optional[str]:
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
    """
    Download all image resources from a list of manifest urls.

    Args:
        manifest_url (str): URL of the IIIF manifest
        target_path (str): Path where to save the images (if None, use img_dir/manifest_id)
        img_dir (str|Path): Path where to save the images (ignored if target_path is set)
        width (int): Width of the images to download (optional)
        height (int): Height of the images to download (optional)
        sleep (float): Time to sleep between each download (default: 0.25s)
        max_dim (int): Maximal height of the images to download (optional)
    """

    def __init__(
        self,
        manifest_url: str,
        target_path: TPath = None,
        img_dir: TPath = None,
        width: int = None,
        height: int = None,
        sleep: float = 0.25,
        max_dim: int=None,
    ):
        """ """
        self.manifest_url = manifest_url
        self.manifest_id = ""  # Prefix to be used for img filenames

        if target_path is None:
            warnings.warn("Using img_dir is deprecated, please use target_path instead", DeprecationWarning)
            if img_dir is None:
                img_dir = IMG_PATH
            target_path = Path(img_dir) / self.get_dir_name()
        self.manifest_dir_path = target_path

        self.size = self.get_formatted_size(width, height)
        self.sleep = sleep
        self.max_dim = max_dim  # Maximal height in px

    def run(self) -> List[Tuple[str, str]]:
        """
        Check the URL and download all images from a IIIF manifest

        If the URL is not valid, raise a ValueError
        """
        manifest = self.check_url()
        return self.download_manifest(manifest)

    def get_dir_name(self):
        return (
            sanitize_str(self.manifest_url).replace("manifest", "").replace("json", "")
        )

    def check_url(self) -> dict:
        """
        Check if the URL is valid and load the manifest

        If the URL is not valid, raise a ValueError
        """
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
                        # Gallica is not accepting more than 5 downloads of >1000px / min after
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

    def save_iiif_img(
        self, img_rscr: dict, i: int, size: str="full", re_download: bool=False
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Save an image from a IIIF resource, at a given size

        If the image is not valid, try to download it at a reduced size

        Args:

            img_rscr (dict): The image resource
            i (int): The index of the image
            size (str): The size of the image to download (default: full)
            re_download (bool): Whether to re-download the image (default: False)

        Returns:

            Tuple (bool, str, str):
                - A boolean indicating if the image was downloaded
                - The name of the image file (None if source raised an error)
                - The URL of the image (None if source raised an error)
        """
        img_name = f"{self.manifest_id}_{i:04d}.jpg"

        img_url = get_id(img_rscr["service"])
        iiif_url = sanitize_url(f"{img_url}/full/{size}/0/default.jpg")

        if (
            glob.glob(os.path.join(self.manifest_dir_path, f"*_{i:04d}.jpg"))
            and not re_download
        ):
            # if the img is already downloaded, don't download it again
            return False, img_name, iiif_url

        with requests.get(iiif_url, stream=True) as response:
            response.raw.decode_content = True
            try:
                img = Image.open(response.raw)
                # img.verify()
            except (UnidentifiedImageError, SyntaxError):
                if size == "full":
                    # Maybe the image is too large, try to download it at a reduced size
                    size = get_reduced_size(img_rscr["width"])
                    return self.save_iiif_img(
                        img_rscr, i, self.get_formatted_size(size)
                    )
                else:
                    console(f"{iiif_url} is not a valid img file")
                    return False, None, None
            except (IOError, OSError):
                if size == "full":
                    # Maybe the image is truncated or corrupted, try to download it at a reduced size
                    size = get_reduced_size(img_rscr["width"])
                    return self.save_iiif_img(
                        img_rscr, i, self.get_formatted_size(size)
                    )
                else:
                    console(f"{iiif_url} is a truncated or corrupted image")
                    return False, None, None

            self.save_img(img, img_name, f"Failed to save {iiif_url}")
        return (True, img_name, iiif_url)

    def save_img(self, img: Image, img_filename: TPath, error_msg: str="Failed to save img") -> bool:
        """
        Save an image to disk
        """
        try:
            img.save(self.manifest_dir_path / img_filename)
            return True
        except Exception:
            console(f"{error_msg}")
        return False

    def get_manifest_id(self, manifest: dict) -> str:
        """
        Get the ID of a IIIF manifest
        """
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
