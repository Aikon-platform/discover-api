import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2

from ...shared.utils.img import get_imgs_in_dirs


def img_id(img_path):
    img_name = img_path.split("/")[-1].split(".")[0]
    return img_name.lower().strip().replace("-", "").replace(",", "-")


class IllusDataset(Dataset):
    def __init__(self, img_dirs, transform=None, device="cpu"):
        self.transform = transform
        self.device = device
        self.img_dirs = img_dirs if type(img_dirs) is list else [img_dirs]
        self.data_paths = self.load_image_paths()

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.data_paths[idx])
        img = torch.from_numpy(img).float().permute(2, 0, 1).to(self.device)
        if self.transform:
            img = self.apply_transform(img)

        return img

    def load_image_paths(self):
        return get_imgs_in_dirs(self.img_dirs)

    def get_image_paths(self):
        # img_ids = [img_id(path) for path in self.data_paths]
        return self.data_paths  # , img_ids

    def apply_transform(self, img):
        transforms_ = []
        for tr in self.transform:
            if tr == "resize":
                transforms_.append(transforms.Resize((224, 224)))
            if tr == "normalize":
                transforms_.append(
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                )
        transforms_ = transforms.Compose(transforms_)
        img = transforms_(img)
        return img
