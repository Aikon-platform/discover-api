from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class FileListDataset(Dataset):
    def __init__(self, data_paths, transform=None, device="cpu"):
        self.transform = transform
        self.device = device
        self.data_paths = data_paths

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        img = transforms.ToTensor()(Image.open(self.data_paths[idx])).to(self.device)
        return self.transform(img)

    def get_image_paths(self):
        return self.data_paths
