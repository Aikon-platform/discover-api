"""
The Dataset class, which represents a dataset of documents
"""

from pathlib import Path
from typing import List, Union, Optional
import json

from ..const import DATASETS_PATH
from .document import Document


class Dataset:
    """
    This class represents a dataset of documents

    :param uid: The unique identifier of the dataset
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

    def __init__(self, uid:str, path: Union[Path, str]=None, documents: Optional[List[str]]=None, load:bool=False):
        """
        Create a new dataset
        """
        self.uid = uid
        if path is None:
            path = DATASETS_PATH / uid
        self.path = Path(path)
        self.documents = [Document(doc) for doc in documents] if documents else None
        if load:
            self.load()

    @property
    def results_path(self) -> Path:
        """
        The path to the results of the dataset
        """
        return self.path / "results"

    def save(self) -> None:
        """
        Save the dataset to disk
        """
        self.path.mkdir(parents=True, exist_ok=True)
        with open(self.path / "documents.json", "w") as f:
            json.dump([doc.uid for doc in self.documents], f)

    def load(self) -> None:
        """
        Load the dataset from disk
        """
        with open(self.path / "documents.json", "r") as f:
            documents = json.load(f)
        self.documents = [Document(doc) for doc in documents]

    def list_images(self) -> List[str]:
        """
        List all images in the dataset, iterating over all documents

        :return: A list of image paths
        """
        images = []
        for document in self.documents:
            images.extend(document.list_images())
        return images