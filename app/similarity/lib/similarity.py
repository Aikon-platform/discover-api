import os
from typing import Optional
import requests
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
import cv2
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader

from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform

from ..const import SCORES_PATH, IMG_PATH
from .const import (
    SEG_STRIDE,
    MAX_SIZE,
    COS_TOPK,
    FEAT_NET,
    FEAT_SET,
    FEAT_LAYER,
)
from .dataset import FileListDataset
from .features import FeatureExtractor
from . import segswap

from .utils import get_model_path, doc_pairs

from ...shared.dataset import Dataset, Document
from ...shared.utils import get_device
from ...shared.utils.fileutils import has_content
from ...shared.utils.img import download_images
from ...shared.utils.logging import LoggingTaskMixin, console, send_update
from ...shared.tasks import LoggedTask

def compute_cosine_similarity(features: np.ndarray, topk: int=COS_TOPK):
    """
    Compute the cosine similarity between all pairs of images in the dataset

    Args:
        features (np.ndarray): The features of the images (n_img, n_feat)
        topk (int): The number of best matches to return

    Returns:
        A list of unique pairs (k_i, k_j, sim) where k_i and k_j are feature indices
    """
    mat = squareform(pdist(features, metric="cosine"))
    tops = np.argsort(mat, axis=1)[:, :topk]
    all_pairs = set(
        (min(i, j), max(i, j), mat[i, j]) for i, row in enumerate(tops) for j in row if i != j
    )
    return sorted(all_pairs)

def _convert_to_pairs(cosine_pairs, image_list: list[str]):
    # Legacy format ? should be basenames but we need to trace documents back ??
    return np.array([(round(sim, 5)*100, image_list[i], image_list[j]) for i, j, sim in cosine_pairs])

@torch.no_grad()
def compute_segswap_similarity(source_images: list[str], pairs: list[tuple[int, int]], device="cuda"):
    """
    Compute the similarity between pairs of images using the SegSwap algorithm

    Args:
        source_images (list[str]): The list of image paths
        pairs (list[tuple[int, int, *any]]): The cosine similarity pairs (i, j, *any)
        output_file (str): The file to save the similarity scores (default: None)
        device (str): The device to run the computation on

    Returns:
        A list of pairs (k_i, k_j, sim)
    """
    param = torch.load(get_model_path("hard_mining_neg5"), map_location=device)
    backbone = segswap.load_backbone(param).to(device)
    encoder = segswap.load_encoder(param).to(device)

    feat_size = MAX_SIZE // SEG_STRIDE
    mask = np.ones((feat_size, feat_size), dtype=bool)
    y_grid, x_grid = np.where(mask)

    batch_size = COS_TOPK
    batched_pairs = [
        pairs[i : i + batch_size] for i in range(0, len(pairs), batch_size)
    ]

    img_dataset = FileListDataset(
        img_dirs=source_images,
        transform=transforms.Compose([
            transforms.Resize((segswap.MAX_SIZE, segswap.MAX_SIZE)), 
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        ),
        device=device,
    )

    out_scores = []
    p_i = None
    # TODO make a dataloader
    for batch in batched_pairs:
        tensor1 = []
        tensor2 = []
        pairs = []
        for i, j, *_ in batch:
            if p_i != i: # Avoid loading several times (cos_pairs are supposed sorted)
                q_tensor = img_dataset[i]
                p_i = i
            
            r_tensor = img_dataset[j]

            tensor1.append(q_tensor)
            tensor2.append(r_tensor)
            pairs.append((i, j))

        scores = segswap.compute_score(
            torch.stack(tensor1),
            torch.stack(tensor2),
            backbone,
            encoder,
            y_grid,
            x_grid,
        )

        out_scores.extend([(pairs[i][0], pairs[i][1], float(score)) for i, score in enumerate(scores)])

    return out_scores

class ComputeSimilarity(LoggedTask):
    """
    Compute the similarity between images inside a dataset
    """
    def __init__(
        self,
        dataset: Dataset,
        parameters: Optional[dict] = None,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.dataset = dataset

        self.feat_net = parameters.get("feat_net", FEAT_NET) if parameters else FEAT_NET
        self.feat_set = parameters.get("feat_set", FEAT_SET) if parameters else FEAT_SET
        self.feat_layer = (
            parameters.get("feat_layer", FEAT_LAYER) if parameters else FEAT_LAYER
        )
        self.topk = parameters.get("topk", COS_TOPK)
        self.algorithm = parameters.get("algorithm", "cosine")
        self.device = get_device()

    @torch.no_grad()
    def get_features(self, source_images: list[str]):
        """
        Extract features from a list of images
        """
        extractor = FeatureExtractor(
            feat_net=self.feat_net,
            feat_set=self.feat_set,
            feat_layer=self.feat_layer,
            device=self.device,
        )

        img_dataset = FileListDataset(
            img_dirs=source_images,
            transform=transforms.Compose([
                transforms.Resize((224, 224)), 
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
            ),
            device=self.device,
        )

        data_loader = DataLoader(img_dataset, batch_size=128, shuffle=False)

        features = extractor.extract_features(data_loader, cache_dir=self.dataset.path / "features")

        if not features.numel():
            self.print_and_log_warning("[task.similarity] No features extracted")
            self.task_update("ERROR", "[API ERROR] No features extracted")
            return
        
        del extractor

        return features

    @torch.no_grad()
    def compute_similarity(self):
        source_images = self.dataset.prepare()
        source_paths = [str(i.path) for i in source_images]
        self.print_and_log(
            f"[task.similarity] {len(source_images)} images downloaded and/or cropped"
        )

        features = self.get_features(source_paths)

        pairs = compute_cosine_similarity(features.cpu().numpy(), topk=self.topk)

        if self.algorithm == "segswap":
            self.print_and_log(
                f"[task.similarity] Computing SegSwap similarity for {len(pairs)} pairs"
            )
            pairs = compute_segswap_similarity(source_paths, pairs, device=self.device)

        self.print_and_log(
            f"[task.similarity] Computed similarity for {len(pairs)} pairs"
        )

        # TODO save and return pairs based on source_image

    def run_task(self):
        if not self.check_dataset():
            self.print_and_log_warning("[task.similarity] No documents to compare")
            self.task_update("ERROR", "[API ERROR] No documents to compare")
            return
        
        self.print_and_log(
            f"[task.similarity] Similarity task triggered for {list(self.dataset.keys())} with {self.feat_net}!"
        )
        self.task_update("STARTED")

        try:
            self.initialize()

            self.compute_similarity()

            self.print_and_log(
                f"[task.similarity] Successfully computed similarity scores"
            )
            self.task_update("SUCCESS")
    
        except Exception as e:
            self.handle_error("Error initializing similarity task", e)
            self.task_update("ERROR", self.error_list)
            return False
        finally:
            self.terminate()

    def check_dataset(self):
        # TODO add more checks
        return len(self.dataset.documents) > 0

    def send_scores(self, doc_pair, score_file):
        if not self.notify_url:
            return
        npy_pairs = {}
        with open(score_file, "rb") as file:
            # Remove client_id prefix from file name
            doc_pair = (
                "_".join(doc_pair[0].split("_")[1:]),
                "_".join(doc_pair[1].split("_")[1:]),
            )
            npy_pairs["-".join(sorted(doc_pair))] = (
                f"{'-'.join(sorted(doc_pair))}.npy",
                file.read(),
            )

            response = requests.post(
                url=f"{self.notify_url}",
                files=npy_pairs,
                data={
                    "experiment_id": self.experiment_id,
                },
            )
            response.raise_for_status()

    def compute_and_send_scores(self):
        for doc_pair in doc_pairs(self.doc_ids):
            score_file = self.compute_scores(doc_pair)
            if not score_file:
                self.print_and_log_warning(
                    f"[task.similarity] Error when computing scores for {doc_pair}"
                )
                continue

            try:
                self.send_scores(doc_pair, score_file)
                self.print_and_log(
                    f"[task.similarity] Successfully send scores for {doc_pair} to {self.notify_url}",
                    color="magenta",
                )
            except requests.exceptions.RequestException as e:
                self.print_and_log(
                    f"[task.similarity] Error in callback request for {doc_pair}", e
                )
                raise Exception
            except Exception as e:
                self.print_and_log(
                    f"[task.similarity] An error occurred for {doc_pair}", e
                )
                raise Exception
        if len(self.computed_pairs) > 0:
            self.print_and_log(
                f"[task.similarity] Successfully computed pairs for {self.computed_pairs}"
            )
        return True

    def compute_scores(self, doc_pair: tuple):
        pair_name = "-".join(sorted(doc_pair))
        score_file = SCORES_PATH / f"{pair_name}.npy"
        if not os.path.exists(score_file):
            self.print_and_log(f"COMPUTING SIMILARITY FOR {doc_pair}", color="magenta")
            success = self.compute_pairs(doc_pair, score_file)
            if success:
                self.computed_pairs.append(pair_name)
                return score_file
            return None
        return score_file

