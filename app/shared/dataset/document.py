from pathlib import Path
import requests
from zipfile import ZipFile
from urllib.parse import urlparse
from typing import List, Optional
import json

from ..const import DOCUMENTS_PATH
from ..utils.iiif import IIIFDownloader, get_json
from ..utils.fileutils import sanitize_str, has_content
from ..utils.img import download_images, MAX_SIZE, download_image
from ..utils.logging import console

class Document:
    """
    A Document is a list of images that are part of a single document
    It corresponds to the "manifest" object in the IIIF Presentation API
    """

    def __init__(self, uid:str, path: Path|str=None):
        self.uid = uid
        if path is None:
            path = DOCUMENTS_PATH / sanitize_str(uid)
        self.path = Path(path)
        self.mapping = {} # A mapping of filenames to their URLs

    @property
    def images_path(self):
        return self.path / "images"
    
    @property
    def cropped_images_path(self):
        return self.path / "cropped"
    
    @property
    def annotations_path(self):
        return self.path / "annotations"
    
    def extend_mapping(self, mapping: dict):
        """
        Extend the mapping of the document with a new mapping
        """
        self.mapping.update(mapping)
        with open(self.path / "mapping.json", "w") as f:
            json.dump(self.mapping, f)
    
    def load_mapping(self):
        """
        Load the mapping of the document from a JSON file
        """
        if not (self.path / "mapping.json").exists():
            return
        with open(self.path / "mapping.json", "r") as f:
            self.mapping = json.load(f)

    def _download_from_iiif(self, manifest_url: str):
        """
        Download images from a IIIF manifest
        """
        self.images_path.mkdir(parents=True, exist_ok=True)
        downloader = IIIFDownloader(manifest_url, img_dir=self.images_path)
        mapping = downloader.run()
        self.extend_mapping(mapping)

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
        for (img_name, img_url) in images_dict.items():
            download_image(img_url, self.images_path, img_name)
        self.extend_mapping(images_dict)

    def download(self, images_src: Optional[str]=None):
        """
        Download a document from its source definition
        """
        if images_src is None:
            images_src = self.uid

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
            console(f"{images_src} format is not handled for a document", color="yellow")

    def list_images(self) -> List[Path]:
        """
        Iterate over the images in the document
        """
        return list(self.images_path.glob("*.jpg"))