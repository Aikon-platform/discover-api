from typing import Optional, Any
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from typing import Iterable
import orjson
from scipy.spatial.distance import pdist, squareform

from .const import SCORES_PATH, MODEL_PATH
from .lib.const import (
    SEG_STRIDE,
    MAX_SIZE,
    COS_TOPK,
    FEAT_NET,
)
from .lib.dataset import FileListDataset
from .lib.features import FeatureExtractor
from .lib import segswap
from .lib.models import get_model_path
from .lib.utils import AllTranspose

from ..shared.dataset import Dataset, Image
from ..shared.utils import get_device
from ..shared.tasks import LoggedTask
from ..shared.utils.logging import serializer


def compute_cosine_similarity(
    features: np.ndarray,
    topk: int = COS_TOPK,
    doc_idx: list[Iterable[int]] = None,
    n_transpositions: int = 1,
    cs_instance=None,
):
    """
    Compute the cosine similarity between all pairs of images in the dataset

    Args:
        features (np.ndarray): The features of the images (n_img, n_feat)
        topk (int): The number of best matches to return
        doc_idx (list[Iterable[int]]): Ranges of indices for each document (default: None)
        n_transpositions (int): features[i:i+n_transpositions] will be considered as referring to the same image
        cs_instance (callable): A callable instance of the class ComputeSimilarity

    Returns:
        A list of unique pairs (k_i, k_j, sim, tr_i, tr_j) where k_i and k_j are feature indices,
        and tr_i and tr_j correspond to the transposition of the best match
    """
    if doc_idx is None:
        doc_idx = [range(len(features))]

    all_mat = squareform(pdist(features, metric="cosine"))
    n_img = features.shape[0]

    if n_transpositions > 1:
        assert n_img % n_transpositions == 0
        n_img //= n_transpositions
        all_mat = (
            all_mat.reshape(n_img, n_transpositions, n_img, n_transpositions)
            .transpose(0, 2, 1, 3)
            .reshape(n_img, n_img, n_transpositions * n_transpositions)
        )
        min_tr = all_mat.argmin(axis=2, keepdims=True)
        all_mat = np.take_along_axis(all_mat, min_tr, axis=2).squeeze(axis=2)
        min_tr_i, min_tr_j = np.divmod(min_tr.squeeze(axis=2), n_transpositions)
    else:
        min_tr_i = np.zeros_like(all_mat)
        min_tr_j = min_tr_i

    np.fill_diagonal(all_mat, 1000)

    all_pairs: set[tuple[int, int, float]] = set()

    # get topk matches for each doc pair
    for doc1 in doc_idx:
        for doc2 in doc_idx:
            mat = all_mat[doc1][:, doc2]
            tops = np.argsort(mat, axis=1)[:, :topk]
            docs_pairs = set(
                (
                    min(doc1[i], doc2[j]),
                    max(doc1[i], doc2[j]),
                    1.0 - mat[i, j],
                    int(min_tr_i[i, j]),
                    int(min_tr_j[i, j]),
                )
                for i, row in enumerate(tops)
                for j in row
                if doc1[i] != doc2[j]
            )
            if cs_instance:
                # MARKER
                cs_instance.notifier(
                    "PROGRESS", output=cs_instance.format_result(docs_pairs)
                )
            all_pairs |= docs_pairs

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
    source_images: list[str], pairs: list[tuple[int, int]], cos_topk, device="cuda"
):
    """
    Compute the similarity between pairs of images using the SegSwap algorithm

    Args:
        source_images (list[str]): The list of image paths
        pairs (list[tuple[int, int, *any]]): The cosine similarity pairs (i, j, *any)
        cos_topk (int): The number of best matches to return
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

    batch_size = cos_topk
    batched_pairs = [
        pairs[i : i + batch_size] for i in range(0, len(pairs), batch_size)
    ]

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
        self.results = {}

        self.dataset = dataset
        self.feat_net = parameters.get("feat_net", FEAT_NET) if parameters else FEAT_NET
        self.topk = parameters.get("topk", COS_TOPK)
        self.algorithm = parameters.get("algorithm", "cosine")

        # Whether to perform pre-filter using cosine similarity to keep only best matches before running segswap
        self.segswap_prefilter = parameters.get("segswap_prefilter", True)
        # If so, how many best matches should be kept (TODO check if it is not cosine_n_filter)
        self.segswap_n = parameters.get("segswap_n", COS_TOPK)

        self.raw_transpositions: list[str] = parameters.get("transpositions", ["none"])
        self.transpositions = [
            getattr(AllTranspose, t.upper()) for t in self.raw_transpositions
        ]

        self.device = get_device()

    @torch.no_grad()
    def get_features(self, source_images: list[str]):
        """
        Extract features from a list of images
        """
        extractor = FeatureExtractor(
            feat_net=self.feat_net,
            device=self.device,
        )

        img_dataset = FileListDataset(
            data_paths=source_images,
            transform=extractor.transforms,
            device=self.device,
            transpositions=self.transpositions,
        )

        data_loader = DataLoader(img_dataset, batch_size=16, shuffle=False)
        cache_id = (
            f"{self.dataset.uid}@{''.join(str(t.value) for t in self.transpositions)}"
        )

        features = extractor.extract_features(
            data_loader,
            cache_dir=self.dataset.path / "features",
            cache_id=cache_id,
        )

        if not features.numel():
            self.print_and_log_warning("[task.similarity] No features extracted")
            self.task_update("ERROR", "[API ERROR] No features extracted")
            return

        del extractor

        return features

    def format_parameters(self):
        return {
            "algorithm": self.algorithm,
            "topk": self.topk,
            "feat_net": self.feat_net,
            "segswap_prefilter": self.segswap_prefilter,
            "segswap_n": self.segswap_n,
            "raw_transpositions": self.raw_transpositions,
            "transpositions": self.transpositions,
        }

    @staticmethod
    def format_pair(pair: tuple) -> tuple[int, int, float, int, int]:
        """
        Format a similarity pair to ensure it has 5 elements, adding default values (0) for missing transpositions.

        Args:
            pair (tuple): A tuple containing either (im_i, im_j, sim) or (im_i, im_j, sim, tr_i, tr_j)

        Returns:
            tuple[int, int, float, int, int]: A 5-element tuple (im_i, im_j, sim, tr_i, tr_j)
        """
        # Pad the pair with zeros to ensure we have 5 elements (for legacy format)
        im_i, im_j, sim, tr_i, tr_j = tuple(list(pair) + [0] * (5 - len(pair)))
        return im_i, im_j, round(float(sim), 4), tr_i, tr_j

    def format_results(
        self, pairs: list[tuple[int, int, float]], source_images: list[Image]
    ) -> dict[
        str,
        dict[str, list[str] | bool | list[Any] | str | Any]
        | dict[str, dict[Any, dict] | list[str] | list[dict[str, Any]]]
        | list[tuple[Any, Any, float, Any, Any]],
    ]:
        """
        Format the results for output

        Args:
            pairs (list[tuple[int, int, float]]): The similarity pairs
            source_images (list[Image]): The source images

        Returns:
            A dictionary with the document index and pairs
        """
        return {
            "parameters": self.format_parameters(),
            "index": {
                "sources": {doc.uid: doc.to_dict() for doc in self.dataset.documents},
                "images": [
                    {**im.to_dict(), "doc_uid": im.document.uid} for im in source_images
                ],
                "transpositions": self.raw_transpositions,
            },
            "pairs": [self.format_pair(pair) for pair in pairs],
        }

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
            features.cpu().numpy(),
            topk=topk,
            doc_idx=source_doc_ranges,
            n_transpositions=len(self.transpositions),
        )

        if self.algorithm == "segswap":
            self.print_and_log(
                f"[task.similarity] Computing SegSwap similarity for {len(pairs)} pairs"
            )
            pairs = compute_segswap_similarity(
                source_paths, pairs, cos_topk=topk, device=self.device
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

        if scores := self.check_already_computed():
            return {
                "dataset_url": self.dataset.get_absolute_url(),
                "annotations": scores,
            }

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
            self.task_update("ERROR", message=self.error_list)
            return False
        finally:
            pass

    def check_parameters(self, parameters):
        """
        Return True if all the parameters are the same (meaning that the similarity has already been computed)
        False if one of the parameters is not the same
        """
        if parameters is None:
            return False
        if parameters.get("algorithm", None) != self.algorithm:
            return False
        if parameters.get("topk", None) != self.topk:
            return False
        if parameters.get("feat_net", None) != self.feat_net:
            return False
        if parameters.get("segswap_n", None) != self.segswap_n:
            return False

        # OTHER PARAMETERS TO CHECK
        # "segswap_prefilter": self.segswap_prefilter,
        # "raw_transpositions": self.raw_transpositions,
        # "transpositions": self.transpositions,
        return True

    def check_already_computed(self):
        # Search through all subdirectories
        for path in SCORES_PATH.rglob(f"{self.dataset.uid}-scores.json"):
            if path.is_file():
                try:
                    scores = orjson.loads(path.read_text())
                    if self.check_parameters(scores.get("parameters", None)):
                        return scores
                except (orjson.JSONDecodeError, OSError) as e:
                    self.print_and_log_warning(
                        f"[task.similarity] Error reading existing scores file {path}: {e}"
                    )
                    continue
        return False

    def check_dataset(self):
        if self.dataset is None:
            return False

        if len(self.dataset.documents) == 0:
            return False

        return True
