"""
The Dataset class, which represents a dataset of documents
"""

from flask import url_for
from pathlib import Path
from typing import List, Union, Optional, Dict
import json

from ... import config
from ..const import DATASETS_PATH
from ..utils.logging import console
from ..utils import hash_str

from .document import Document


class Dataset:
    """
    This class represents a dataset of documents

    :param uid: The unique identifier of the dataset (if not provided, generate one)
    :param path: The path to the dataset on disk (default: DATASETS_PATH/uid)
    :param documents: The list of documents in the dataset (if not provided, load from disk)
    :param load: Whether to load the dataset from disk (default: False)

    The dataset is saved to disk in the following structure:

    .. code-block:: none

        - path/
            - documents.json
            - results/
                - ...
    """

    def __init__(self, uid:str=None, path: Union[Path, str]=None, documents: Optional[List[dict]]=None, load:bool=False):
        """
        Create a new dataset
        """
        if isinstance(documents, dict): # legacy format
            documents = [{"uid": uid, "src": src, "type": "url_list"} for uid, src in documents.items()]

        if uid is None:
            uid = hash_str("".join([doc["src"] for doc in documents]))

        self.uid = uid

        if path is None:
            path = DATASETS_PATH / uid
        self.path = Path(path)

        # TODO Check consistency when reusing a dataset

        if load:
            self.load()
        else:
            self.documents = [Document.from_dict(doc) for doc in documents] if documents else None

    def to_dict(self, with_url: bool=False) -> Dict:
        """
        Convert the dataset to a dictionary
        """
        ret = {
            "uid": self.uid,
            "documents": [doc.to_dict(with_url) for doc in self.documents],
        }
        if with_url:
            ret["url"] = self.get_absolute_url()
        return ret

    @property
    def results_path(self) -> Path:
        """
        The path to the results of the dataset
        """
        return self.path / "results"
    
    def get_absolute_url(self) -> str:
        """
        Get the absolute URL of the dataset
        """
        return f"{config.BASE_URL}/datasets/dataset/{self.uid}"
        return url_for("datasets.dataset_info", uid=self.uid, _external=True)

    def save(self) -> None:
        """
        Save the dataset to disk
        """
        self.path.mkdir(parents=True, exist_ok=True)
        with open(self.path / "info.json", "w") as f:
            # json.dump([doc.uid for doc in self.documents], f)
            json.dump(self.to_dict(), f)

    def load(self) -> None:
        """
        Load the dataset from disk
        """
        with open(self.path / "info.json", "r") as f:
            documents = json.load(f)["documents"]
        self.documents = [Document.from_dict(doc) for doc in documents]

    def list_images(self) -> List[str]:
        """
        List all images in the dataset, iterating over all documents

        :return: A list of image paths
        """
        images = []
        for document in self.documents:
            images.extend(document.list_images())
        return images
