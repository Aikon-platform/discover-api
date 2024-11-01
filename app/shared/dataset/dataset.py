from pathlib import Path
from typing import List, Union, Optional, Dict
import json

from ..const import DATASETS_PATH
from .document import Document
from ..utils.logging import console


class Dataset:  # Discover-demo Dataset
    def __init__(
        self,
        uid: str,
        path: Union[Path, str] = None,
        documents=None,
        load: bool = False
    ):
        self.uid = uid
        if path is None:
            path = DATASETS_PATH / uid
        self.path = Path(path)
        if load:
            self.load()
        else:
            console(documents, color="red")
            self.documents = [Document(uid, src=src) for uid, src in documents.items()] if documents else None

    @property
    def results_path(self):
        return self.path / "results"

    def save(self):
        """
        Save the dataset to disk
        """
        self.path.mkdir(parents=True, exist_ok=True)
        with open(self.path / "documents.json", "w") as f:
            # json.dump([doc.uid for doc in self.documents], f)
            json.dump({doc.uid: doc.src for doc in self.documents}, f)

    def load(self):
        """
        Load the dataset from disk
        """
        with open(self.path / "documents.json", "r") as f:
            documents = json.load(f)
        self.documents = [Document(uid, src=src) for uid, src in documents.items()]

    def list_images(self):
        """
        List all images in the dataset
        """
        images = []
        for document in self.documents:
            images.extend(document.list_images())
        return images
