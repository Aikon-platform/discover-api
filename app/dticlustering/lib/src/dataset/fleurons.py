from functools import lru_cache
from PIL import Image
import os

import pandas as pd
from torch.utils.data.dataset import Dataset as TorchDataset
from torchvision.transforms import Compose, Resize, ToTensor

from ..utils import coerce_to_path_and_check_exist
from ..utils.path import DATASETS_PATH


class FleuronsDataset(TorchDataset):
    root = DATASETS_PATH
    name = "fleurons"
    n_channels = 3

    def __init__(self, split=None, **kwargs) -> None:
        self.data_path = coerce_to_path_and_check_exist(self.root / "fleurons")
        self.split = None
        self.n_classes = kwargs.get("n_classes", 36)

        self.img_size = kwargs.get("img_size", 128)
        if type(self.img_size) is int:
            self.img_size = (self.img_size, self.img_size)
        elif len(self.img_size) == 2:
            self.img_size = self.img_size
        else:
            raise ValueError

        try:
            metadata_path = os.path.join(
                self.data_path, kwargs.get("annotation_path", "metadata.tsv")
            )
            annotations = pd.read_csv(metadata_path, delimiter="\t")
            img_path = os.path.join(self.data_path, kwargs.get("img_path", "imgs"))
            self.input_files = [
                os.path.join(img_path, filename)
                for filename in annotations["image_filename"]
            ]
            self.labels, _ = pd.factorize(annotations["motif_id"])
        except FileNotFoundError:
            self.files = []

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        img = self.transform(Image.open(self.input_files[idx]).convert("RGB"))
        return img, self.labels[idx], [], str(self.input_files[idx])

    @property
    @lru_cache()
    def transform(self):
        return Compose([Resize(self.img_size), ToTensor()])
