"""
The Document class, which represents a document in the dataset
"""

from flask import url_for

import requests
import json
import httpx
from pathlib import Path
from PIL import Image as PImage
from stream_unzip import stream_unzip
from typing import List, Optional
from iiif_download import IIIFManifest

from ... import config
from ..const import DOCUMENTS_PATH
from ..utils.fileutils import sanitize_str, check_if_file
from ..utils.img import MAX_SIZE, download_image, get_img_paths, get_json
from ..utils.logging import console, serializer
from .utils import Image, pdf_to_img


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".json", ".tiff", ".pdf"}


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
            - images.json
    """

    def __init__(
        self,
        uid: str = None,
        dtype: str = "zip",
        path: Path | str = None,
        src: Optional[str] = None,
    ):
        self.uid = sanitize_str(uid if uid is not None else src)
        self.path = Path(path if path is not None else DOCUMENTS_PATH / dtype / uid)
        self.src = src
        self.dtype = dtype
        self._images = []

    @classmethod
    def from_dict(cls, doc_dict: dict) -> "Document":
        """
        Create a new Document from a dictionary
        """
        return Document(
            doc_dict.get("uid", None), doc_dict["type"], src=doc_dict["src"]
        )

    def to_dict(self, with_url: bool = False) -> dict:
        """
        Convert the document to a dictionary
        """
        ret = {
            "uid": self.uid,
            "type": self.dtype,
            "src": str(self.src),
        }
        if with_url:
            ret["url"] = self.get_absolute_url()
            ret["download"] = self.get_download_url()
        return ret

    def get_absolute_url(self):
        """
        Get the absolute URL of the document
        """
        return f"{config.BASE_URL}/datasets/document/{self.dtype}/{self.uid}"
        # return url_for("datasets.document_info", dtype=self.dtype, uid=self.uid, _external=True)

    def get_download_url(self):
        """
        Get the URL to download the document
        """
        return f"{self.get_absolute_url()}/download"
        # return url_for("datasets.document_download", dtype=self.dtype, uid=self.uid, _external=True)

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
    def images_info_path(self):
        return self.path / "images.json"

    @property
    def images(self):
        """
        List of images in the document
        """
        if len(self._images) == 0:
            self.load_images()
        return self._images

    def save_images(self, images: List[Image]):
        """
        Save the images of the document
        """
        self._images = images
        with open(self.images_info_path, "w") as f:
            json.dump([img.to_dict() for img in images], f, default=serializer)

    def load_images(self):
        """
        Load the images of the document
        """
        if not self.images_info_path.exists():
            # TODO save it? or regenerate on each load?
            self._images = self.list_images_from_path()
            return
        with open(self.images_info_path, "r") as f:
            self._images = [Image.from_dict(img, self) for img in json.load(f)]

    def _download_from_iiif(self, manifest_url: str):
        """
        Download images from a IIIF manifest
        """
        manifest = IIIFManifest(manifest_url)
        manifest.download(save_dir=self.images_path)
        self.save_images(
            [
                Image(
                    id=iiif_image.img_name,
                    src=iiif_image.url,
                    path=iiif_image.img_path,
                    metadata={"page": iiif_image.idx},
                    document=self,
                )
                for iiif_image in manifest.get_images()
            ]
        )

    def _download_from_zip(self, zip_url: str):
        """
        Download a zip file from a URL, extract its contents, and save images.
        """

        def zipped_chunks():
            with httpx.stream("GET", zip_url) as r:
                yield from r.iter_bytes(chunk_size=8192)
            # with requests.get(zip_url, stream=True) as r:
            #     r.raise_for_status()
            #     for chunk in r.iter_content(chunk_size=8192):
            #         yield chunk

        def skip():
            for _ in unzipped_chunks:
                pass

        for file_name, file_size, unzipped_chunks in stream_unzip(zipped_chunks()):
            file_name = file_name.decode("utf-8")
            if "/." in "/" + file_name.replace("\\", "/"):  # hidden file
                skip()
                continue
            path = self.images_path / file_name
            if path.suffix.lower() not in ALLOWED_EXTENSIONS:
                skip()
                continue

            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as f:
                for chunk in unzipped_chunks:
                    f.write(chunk)

    def _download_from_url_list(self, images_list_url: str):
        """
        Download images from a dictionary of URLs [img_stem -> img_url]
        """
        images_dict = get_json(images_list_url)
        images = []
        for img_stem, img_url in images_dict.items():
            download_image(img_url, self.images_path, img_stem)
            images.append(
                Image(
                    id=img_stem,
                    src=img_url,
                    path=self.images_path / f"{img_stem}.jpg",
                    document=self,
                )
            )
        self.save_images(images)

    def _download_from_pdf(self, pdf_url: str):
        """
        Download pdf, convert to images and save
        """
        pdf_path = self.path / pdf_url.split("/")[-1]
        with requests.get(pdf_url, stream=True) as r:
            r.raise_for_status()
            with open(pdf_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        pdf_to_img(pdf_path, self.images_path)
        pdf_path.unlink()

    def _download_from_img(self, img_url: str):
        """
        Download image and save
        """
        # TODO make this work
        download_image(img_url, self.images_path, "image_name")

    def download(self) -> None:
        """
        Download a document from its source definition
        """
        console(f"Downloading [{self.dtype}] {self.uid}...", color="blue")

        self.images_path.mkdir(parents=True, exist_ok=True)

        if self.dtype == "iiif":
            self._download_from_iiif(self.src)
        elif self.dtype == "zip":
            self._download_from_zip(self.src)
        elif self.dtype == "url_list":
            self._download_from_url_list(self.src)
        elif self.dtype == "pdf":
            self._download_from_pdf(self.src)
        elif self.dtype == "img":
            self._download_from_img(self.src)
        else:
            raise ValueError(f"Unknown document type: {self.dtype}")

    def list_images(self) -> List[Image]:
        """
        Iterate over the images in the document
        """
        return self.images

    def list_images_from_path(self) -> List[Image]:
        """
        Iterate over the images in the document's folder
        """
        return [
            Image(
                id=str(img_path.relative_to(self.images_path)),
                src=str(img_path.relative_to(self.path)),
                path=img_path,
                document=self,
            )
            for img_path in get_img_paths(self.images_path)
        ]

    def has_images(self) -> bool:
        return check_if_file(self.images_path, extensions=ALLOWED_EXTENSIONS)

    def prepare_crops(self, crops: List[dict]) -> List[Image]:
        """
        Prepare crops for the document

        Args:
            crops: A list of crops {document, source, source_info, crops: [{crop_id, relative: {x1, y1, w, h}}]}

        Returns:
            A list of Image objects
        """
        source = None
        im = None
        crop_list = []

        mapping = {im.id: im for im in self.images}

        for img in crops:
            if img["doc_uid"] != self.uid:
                continue
            source_info = (
                Image.from_dict(img["source_info"], self)
                if img.get("source_info")
                else None
            )

            for crop in img["crops"]:
                crop_path = self.cropped_images_path / f"{crop['crop_id']}.jpg"
                crop_sum = ",".join(
                    [f'{crop["relative"][k]:0.3f}' for k in ["x1", "y1", "x2", "y2"]]
                )

                crop_list.append(
                    Image(
                        id=crop["crop_id"],
                        src=getattr(source_info, 'src', None) or crop_path.name,
                        path=crop_path,
                        metadata={
                            **(getattr(source_info, 'metadata', None) or {}),
                            "crop": crop_sum,
                        },
                        document=self,
                    )
                )
                if crop_path.exists():
                    continue

                crop_path.parent.mkdir(parents=True, exist_ok=True)

                if source != img["source"]:
                    source = img["source"]
                    if source in mapping:
                        im = mapping[source].path
                    else:
                        im = self.images_path / source
                    im = PImage.open(im).convert("RGB")

                box = crop["relative"]
                if "x1" in box:
                    x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
                else:
                    x1, y1, w, h = box["x"], box["y"], box["width"], box["height"]
                    x2, y2 = x1 + w, y1 + h
                x1, y1, x2, y2 = (
                    int(x1 * im.width),
                    int(y1 * im.height),
                    int(x2 * im.width),
                    int(y2 * im.height),
                )
                if x2 - x1 == 0 or y2 - y1 == 0:
                    # use placeholder image
                    im_cropped = PImage.new("RGB", (MAX_SIZE, MAX_SIZE))
                else:
                    im_cropped = im.crop((x1, y1, x2, y2))
                im_cropped.save(crop_path)

        return crop_list
