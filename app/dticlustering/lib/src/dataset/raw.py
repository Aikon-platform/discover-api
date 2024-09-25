from abc import ABCMeta
from functools import lru_cache
from PIL import Image

from torch.utils.data.dataset import Dataset as TorchDataset
from torchvision.transforms import CenterCrop, Compose, Resize, ToTensor

from ..utils import coerce_to_path_and_check_exist, get_files_from_dir
from ..utils.image import IMG_EXTENSIONS
from ..utils.path import DATASETS_PATH
from pathlib import Path


class _AbstractCollectionDataset(TorchDataset):
    """Abstract torch dataset from raw files collections associated to tags."""

    __metaclass__ = ABCMeta
    root = DATASETS_PATH
    name = NotImplementedError
    n_channels = 3
    include_recursive = False

    def __init__(self, split, img_size, **kwargs):
        tag = kwargs.get("tag", "")
        self.data_path = (
            coerce_to_path_and_check_exist(self.root / self.name / tag) / split
        )
        self.split = split
        try:
            input_files = get_files_from_dir(
                self.data_path,
                IMG_EXTENSIONS,
                sort=True,
                recursive=self.include_recursive,
            )
            input_files = [p for p in input_files if not "/__" in str(p)]
        except FileNotFoundError:
            input_files = []

        self.input_files = input_files
        self.labels = [-1] * len(input_files)
        self.n_classes = 0
        self.size = len(self.input_files)

        if isinstance(img_size, int):
            self.img_size = (img_size, img_size)
            self.crop = True
        else:
            assert len(img_size) == 2
            self.img_size = img_size
            self.crop = False

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img = Image.open(self.input_files[idx])
        if img.mode == "RGBA":
            alpha = img.split()[-1]
        else:
            h, w = img.size
            alpha = Image.new("L", (h, w), (255))
        inp = self.transform(img.convert("RGB"))
        alpha = self.transform(alpha)
        return inp, self.labels[idx], alpha, str(self.input_files[idx])

    @property
    @lru_cache()
    def transform(self):
        if self.crop:
            size = self.img_size[0]
            transform = [Resize(size), CenterCrop(size), ToTensor()]
        else:
            transform = [Resize(self.img_size), ToTensor()]
        return Compose(transform)


class MegaDepthDataset(_AbstractCollectionDataset):
    name = "megadepth"


class GenericDataset(_AbstractCollectionDataset):
    name = "generic"
    include_recursive = True


class LettersDataset(_AbstractCollectionDataset):
    name = "Lettre_e"


class CoADataset(_AbstractCollectionDataset):
    name = "coa_marion"
