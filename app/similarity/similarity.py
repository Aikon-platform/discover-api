from typing import Optional
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from typing import Iterable
import orjson

from scipy.spatial.distance import pdist, squareform

from .const import SCORES_PATH
from .lib.const import (
    SEG_STRIDE,
    MAX_SIZE,
    COS_TOPK,
    FEAT_NET,
    FEAT_SET,
    FEAT_LAYER,
)
from .lib.dataset import FileListDataset
from .lib.features import FeatureExtractor
from .lib import segswap

from .lib.utils import get_model_path

from ..shared.dataset import Dataset, Image
from ..shared.utils import get_device
from ..shared.tasks import LoggedTask
from ..shared.utils.logging import serializer


def compute_cosine_similarity(
    features: np.ndarray, topk: int = COS_TOPK, groups: list[Iterable[int]] = None
):
    """
    Compute the cosine similarity between all pairs of images in the dataset

    Args:
        features (np.ndarray): The features of the images (n_img, n_feat)
        topk (int): The number of best matches to return
        groups (list[Iterable[int]]): Ranges of indices for each document (default: None)

    Returns:
        A list of unique pairs (k_i, k_j, sim) where k_i and k_j are feature indices
    """
    if groups is None:
        groups = [range(len(features))]

    all_mat = squareform(pdist(features, metric="cosine"))
    np.fill_diagonal(all_mat, 1000)

    all_pairs: set[tuple[int, int, float]] = set()

    # get topk matches for each group pairs
    for group1 in groups:
        for group2 in groups:
            mat = all_mat[group1][:, group2]
            tops = np.argsort(mat, axis=1)[:, :topk]
            all_pairs |= set(
                (min(group1[i], group2[j]), max(group1[i], group2[j]), 1.0 - mat[i, j])
                for i, row in enumerate(tops)
                for j in row
                if group1[i] != group2[j]
            )

    return sorted(all_pairs)


def _convert_to_pairs(cosine_pairs, image_list: list[str]):
    # Legacy format? should be basenames, but we need to trace documents back??
    return np.array(
        [
            (round(sim, 5) * 100, image_list[i], image_list[j])
            for i, j, sim in cosine_pairs
        ]
    )


@torch.no_grad()
def compute_segswap_similarity(
    source_images: list[str], pairs: list[tuple[int, int]], topk, device="cuda"
):
    """
    Compute the similarity between pairs of images using the SegSwap algorithm

    Args:
        source_images (list[str]): The list of image paths
        pairs (list[tuple[int, int, *any]]): The cosine similarity pairs (i, j, *any)
        topk (int): The number of best matches to return
                    NOTE: for now, this is useless because topk is used before for filtering out cosine worse matches
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

    batched_pairs = [pairs[i : i + topk] for i in range(0, len(pairs), topk)]

    img_dataset = FileListDataset(
        data_paths=source_images,
        transform=transforms.Compose(
            [
                transforms.Resize((segswap.MAX_SIZE, segswap.MAX_SIZE)),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
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
            if p_i != i:  # Avoid loading several times (cos_pairs are supposed sorted)
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

        out_scores.extend(
            [(pairs[i][0], pairs[i][1], float(score)) for i, score in enumerate(scores)]
        )

    return out_scores


def _make_diff_ranges(items: list) -> list[range]:
    """
    Make a list of ranges of constant values in a list
    """
    ranges = []
    p = 0
    for k, i in enumerate(items):
        if i != items[p]:
            ranges.append(range(p, k))
            p = k
    ranges.append(range(p, len(items)))
    return ranges


class ComputeSimilarity(LoggedTask):
    """
    Compute the similarity between images inside a dataset
    """

    def __init__(
        self, dataset: Dataset, parameters: Optional[dict] = None, *args, **kwargs
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

        # Whether to perform pre-filter using cosine similarity to keep only best matches before running segswap
        self.segswap_prefilter = parameters.get("segswap_prefilter", True)
        # If so, how many best matches should be kept (TODO check if it is not cosine_n_filter)
        self.segswap_n = parameters.get("segswap_n", COS_TOPK)

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
            data_paths=source_images,
            transform=transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            ),
            device=self.device,
        )

        data_loader = DataLoader(img_dataset, batch_size=128, shuffle=False)

        features = extractor.extract_features(
            data_loader,
            cache_dir=self.dataset.path / "features",
            cache_id=self.dataset.uid,
        )

        if not features.numel():
            self.print_and_log_warning("[task.similarity] No features extracted")
            self.task_update("ERROR", "[API ERROR] No features extracted")
            return

        del extractor

        return features

    def format_results_per_doc(
        self, pairs: list[tuple[int, int, float]], source_images: list[Image]
    ) -> list[dict]:
        """
        Format the results for output

        Args:
            pairs (list[tuple[int, int, float]]): The similarity pairs
            source_images (list[Image]): The source images

        Returns:
            A list of dictionaries {doc1: ..., doc2: ..., pairs: [(id1, id2, sim)]}
        """
        # NOT USED NOW
        per_doc_pairs = {}
        for (i, j, sim) in pairs:
            assert i <= j  # avoid duplicates
            im_i = source_images[i]
            im_j = source_images[j]
            doc_i = im_i.document
            doc_j = im_j.document
            key = (doc_i.uid, doc_j.uid)
            if key not in per_doc_pairs:
                per_doc_pairs[key] = []
            per_doc_pairs[key].append((im_i.id, im_j.id, round(float(sim)), 4))

        output_json = [
            {
                "doc1": doc_i,
                "doc2": doc_j,
                "pairs": pairs,
            }
            for (doc_i, doc_j), pairs in per_doc_pairs.items()
        ]

        return output_json

    def format_results(
        self, pairs: list[tuple[int, int, float]], source_images: list[Image]
    ) -> list[dict]:
        """
        Format the results for output

        Args:
            pairs (list[tuple[int, int, float]]): The similarity pairs
            source_images (list[Image]): The source images

        Returns:
            A dictionary with the document index and pairs
        """
        output_json = {
            "index": {
                "sources": {doc.uid: doc.to_dict() for doc in self.dataset.documents},
                "images": [
                    {"id": im.id, "src": im.src, "doc_uid": im.document.uid}
                    for im in source_images
                ],
            },
            "pairs": [(im_i, im_j, round(float(sim), 4)) for im_i, im_j, sim in pairs],
        }

        return output_json

    @torch.no_grad()
    def compute_similarity(self) -> list[dict]:
        """
        Compute the similarity between images in the dataset and returns the results
        """
        source_images = self.dataset.prepare()
        source_paths = [str(i.path) for i in source_images]
        source_doc_ranges = _make_diff_ranges([i.document for i in source_images])

        self.print_and_log(
            f"[task.similarity] {len(source_images)} images downloaded and/or cropped"
        )

        topk = self.segswap_n if self.algorithm == "segswap" else self.topk

        features = self.get_features(source_paths)

        # TODO skip this step if self.algorithm == "segswap" && self.segswap_prefilter == false
        pairs = compute_cosine_similarity(
            features.cpu().numpy(), topk=topk, groups=source_doc_ranges
        )

        if self.algorithm == "segswap":
            self.print_and_log(
                f"[task.similarity] Computing SegSwap similarity for {len(pairs)} pairs"
            )
            pairs = compute_segswap_similarity(
                source_paths, pairs, topk=topk, device=self.device
            )

        self.print_and_log(
            f"[task.similarity] Computed similarity for {len(pairs)} pairs"
        )

        return self.format_results(pairs, source_images)

    def run_task(self):
        if not self.check_dataset():
            self.print_and_log_warning("[task.similarity] No documents to compare")
            self.task_update("ERROR", "[API ERROR] No documents to compare")
            return

        self.task_update("STARTED")
        self.print_and_log(
            f"[task.similarity] Similarity task triggered for {self.dataset.uid} with {self.feat_net}!"
        )

        try:
            similarity = self.compute_similarity()
            self.results = similarity

            tfile = SCORES_PATH / self.experiment_id / f"{self.dataset.uid}-scores.json"
            tfile.parent.mkdir(parents=True, exist_ok=True)

            with open(tfile, "wb") as f:
                f.write(orjson.dumps(similarity, default=serializer))

            self.print_and_log(
                f"[task.similarity] Successfully computed similarity scores"
            )
            self.task_update("SUCCESS")
            return True
        except Exception as e:
            self.handle_error("Error initializing similarity task", e)
            self.task_update("ERROR", self.error_list)
            return False
        finally:
            pass

    def check_dataset(self):
        # TODO add more checks
        return len(self.dataset.documents) > 0
