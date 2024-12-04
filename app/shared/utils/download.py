"""
Download a dataset from front (deprecated?)
"""
import warnings
from pathlib import Path

import requests
from zipfile import ZipFile
from urllib.parse import urlparse

from .iiif import IIIFDownloader
from .img import download_images, MAX_SIZE
from .fileutils import has_content, sanitize_str
from .logging import console
from ..const import IMG_PATH


def download_dataset(dataset_src, datasets_dir_path=None, dataset_dir_name=None, sub_dir=None, dataset_ref=None):
    """
    Download a dataset from front
    
    TODO improve this function / Use dataset.documents.Document.download()
    """
    warnings.warn(
        "download_dataset is deprecated, use Document.download() instead",
        DeprecationWarning,
        stacklevel=2,
    )

    if not datasets_dir_path:
        datasets_dir_path = IMG_PATH

    if not isinstance(datasets_dir_path, Path):
        datasets_dir_path = Path(datasets_dir_path)

    if not dataset_dir_name:
        # TODO improve this
        dataset_dir_name = sanitize_str(dataset_ref or dataset_src).replace("manifest", "").replace("json", "")

    dataset_path = datasets_dir_path / dataset_dir_name

    # if dataset_src is a URL
    if all([urlparse(dataset_src).scheme, urlparse(dataset_src).netloc]):
        # TODO improve check on URL type
        if dataset_src.endswith(".zip"):
            # ZIP FILE
            dataset_path.mkdir(parents=True, exist_ok=True)
            dataset_zip_path = dataset_path / "dataset.zip"

            with requests.get(dataset_src, stream=True) as r:
                r.raise_for_status()
                with open(dataset_zip_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

            with ZipFile(dataset_zip_path, "r") as zipObj:
                dataset_path = dataset_path / sub_dir if sub_dir else dataset_path
                zipObj.extractall(dataset_path)

            dataset_zip_path.unlink()
        else:
            # IIIF MANIFEST
            downloader = IIIFDownloader(dataset_src, img_dir=datasets_dir_path)
            downloader.run()
            return downloader.get_dir_name(), downloader.manifest_id

    elif type(dataset_src) is dict:
        # LIST OF URLS
        doc_ids = []
        for doc_id, url in dataset_src.items():
            try:
                doc_id = f"{dataset_ref}_{doc_id}"
                doc_ids.append(doc_id)
                if not has_content(f"{datasets_dir_path}/{doc_id}/"):
                    download_images(url, doc_id, datasets_dir_path, MAX_SIZE)
            except Exception as e:
                raise ImportError(f"Error downloading images: {e}")
        return datasets_dir_path, dataset_ref, doc_ids
    else:
        console(f"{dataset_src} format is not handled for a dataset", color="yellow")

    return dataset_dir_name, dataset_ref or dataset_dir_name
