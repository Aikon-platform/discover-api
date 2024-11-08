"""
The Document class, which represents a document in the dataset
"""

from pathlib import Path
import requests
from zipfile import ZipFile
from urllib.parse import urlparse
from typing import List, Optional
import json

from ..const import DOCUMENTS_PATH
from ..utils.iiif import IIIFDownloader, get_json
from ..utils.fileutils import sanitize_str, has_content
from ..utils.img import download_images, MAX_SIZE, download_image, get_img_paths
from ..utils.logging import console


class Document:
    """
    A Document is a list of images that are part of a single document

    It can be :
    - downloaded from a single IIIF manifest
    - downloaded from a ZIP file
    - downloaded from a dictionary of single image URLs

    :param uid: The unique identifier of the document
    :param path: The path to the document on disk (default: DOCUMENTS_PATH/uid)

    The document is saved to disk in the following structure:

    .. code-block:: none

        - path/
            - images/
                - ...jpg
            - cropped/
                - ...jpg
            - annotations/
                - ...json
            - mapping.json
    """

    def __init__(self, uid: str, path: Path | str = None, src: Optional[str] = None):
        self.uid = uid
        if path is None:
            path = DOCUMENTS_PATH / sanitize_str(uid)
        self.path = Path(path)
        self.src = src
        self._mapping = None  # A mapping of filenames to their URLs

    @property
    def images_path(self):
        return self.path / "images"

    @property
    def cropped_images_path(self):
        return self.path / "cropped"

    @property
    def annotations_path(self):
        return self.path / "annotations"
    
    @property
    def mapping(self):
        """A mapping of filenames to their URLs"""
        if self._mapping is None:
            self._load_mapping()
        return self._mapping

    def _extend_mapping(self, mapping: dict):
        """
        Extend the mapping of the document with a new mapping
        """
        self.mapping.update(mapping)
        with open(self.path / "mapping.json", "w") as f:
            json.dump(self.mapping, f)

    def _load_mapping(self):
        """
        Load the mapping of the document from a JSON file
        """
        if not (self.path / "mapping.json").exists():
            self._mapping = {}
            return
        with open(self.path / "mapping.json", "r") as f:
            self._mapping = json.load(f)

    def _download_from_iiif(self, manifest_url: str):
        """
        Download images from a IIIF manifest
        """
        self.images_path.mkdir(parents=True, exist_ok=True)
        downloader = IIIFDownloader(manifest_url, img_dir=self.images_path)
        mapping = downloader.run()
        self._extend_mapping(mapping)

    def _download_from_zip(self, zip_url: str):
        """
        Download images from a ZIP file
        """
        self.images_path.mkdir(parents=True, exist_ok=True)
        zip_path = self.path / "dataset.zip"

        with requests.get(zip_url, stream=True) as r:
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        with ZipFile(zip_path, "r") as zipObj:
            zipObj.extractall(self.images_path)

        zip_path.unlink()

    def _download_from_urls(self, images_dict: dict):
        """
        Download images from a dictionary of URLs [img_name -> img_url]
        """
        self.images_path.mkdir(parents=True, exist_ok=True)
        for img_name, img_url in images_dict.items():
            download_image(img_url, self.images_path, img_name)
        self._extend_mapping(images_dict)

    def download(self, images_src: Optional[str] = None) -> None:
        """
        Download a document from its source definition

        :param images_src: The source of the images (IIIF manifest, ZIP file, dictionary of URLs). If None, use the document's UID

        Behavior:

        - If images_src is a .zip URL, download the images from the ZIP file (no mapping)
        - If images_src is a IIIF manifest URL, download the images from the IIIF manifest 
          (mapping is then IIIF image ID -> local image path)
        - If images_src is a dictionary of URLs, or refer to a JSON file containing a dictionary of URLs, 
          download the images from the URLs (mapping is then dict key -> local image path)
        """
        console(self.src, color="green")
        if images_src is None:
            images_src = self.src

        if all([urlparse(images_src).scheme, urlparse(images_src).netloc]):
            if images_src.endswith(".zip"):
                self._download_from_zip(images_src)
            else:
                try:
                    self._download_from_iiif(images_src)
                except ValueError as e:
                    # Not a IIIF manifest, probably a JSON file, let's treat it as a dictionary of URLs
                    return self.download(get_json(images_src))
        elif type(images_src) is dict:
            self._download_from_urls(images_src)
        else:
            console(
                f"{images_src} format is not handled for a document", color="yellow"
            )

    def list_images(self) -> List[Path]:
        """
        Iterate over the images in the document
        """
        # return list(self.images_path.glob("*.jpg"))
        return get_img_paths(self.images_path)
