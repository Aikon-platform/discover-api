from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from .utils import AllTranspose


class FileListDataset(Dataset):
    def __init__(
        self,
        data_paths,
        transform=None,
        device="cpu",
        transpositions: list[AllTranspose] = [AllTranspose.NONE],
    ):
        self.transform = transform
        self.device = device
        self.data_paths = data_paths
        self.rotations = transpositions

    def __len__(self):
        return len(self.data_paths) * len(self.rotations)

    def __getitem__(self, idx):
        # TODO here prevent UnidentifiedImageError
        idx, rot = divmod(idx, len(self.rotations))
        im = Image.open(self.data_paths[idx])

        rot = self.rotations[rot]
        if rot != AllTranspose.NONE:
            im = im.transpose(rot.value)

        # ToTensor is done in transform pipeline
        # img = transforms.ToTensor()(im).to(self.device)

        img = self.transform(im)
        return img.to(self.device)

    def get_image_paths(self):
        return self.data_paths
