from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, TypedDict, Dict, List, TypeAlias, Tuple, cast, Set, TypeVar
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from typing import Iterable
import orjson
from scipy.spatial.distance import pdist, squareform

from .const import SCORES_PATH, DEMO_NAME
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

from ..shared.dataset import Dataset
from ..shared.dataset.document import DocDict, get_file_url
from ..shared.dataset.utils import ImageDict, Image
from ..shared.utils import get_device
from ..shared.tasks import LoggedTask
from ..shared.utils.logging import serializer


PairTuple: TypeAlias = Tuple[int, int, float, int, int]
PairList: TypeAlias = Set[PairTuple] | List[PairTuple]
DocRef: TypeAlias = Tuple[str, str]
DocPairs: TypeAlias = Dict[DocRef, PairList]


@dataclass
class Pair:
    im_i: int
    im_j: int
    similarity: float
    trans_i: int = 0
    trans_j: int = 0

    def to_tuple(self) -> PairTuple:
        return self.im_i, self.im_j, self.similarity, self.trans_i, self.trans_j

    @classmethod
    def from_tuple(cls, data: tuple) -> "Pair":
        values = list(data) + [0] * (5 - len(data))
        return cls(
            im_i=values[0],
            im_j=values[1],
            similarity=round(float(values[2]), 4),
            trans_i=values[3],
            trans_j=values[4],
        )


@dataclass
class DocIndex(TypedDict):
    sources: Dict[str, DocDict]
    images: List[ImageDict]
    transpositions: List[str]


@dataclass
class SimParameters(TypedDict):
    algorithm: str
    topk: int
    feat_net: str
    segswap_prefilter: bool
    segswap_n: Optional[int]
    raw_transpositions: Optional[List[str]]
    transpositions: Optional[List[str]]


@dataclass
class SimilarityResults(TypedDict):
    parameters: SimParameters
    index: DocIndex
    pairs: List[PairTuple]


@dataclass
class DocumentPairs:
    doc1_uid: str
    doc2_uid: str
    pairs: List[PairTuple]


def _make_diff_ranges(docs: list) -> List[range]:
    """
    Make a list of ranges of constant values in a list
    """
    ranges = []
    p = 0
    for k, i in enumerate(docs):
        if i != docs[p]:
            ranges.append(range(p, k))
            p = k
    ranges.append(range(p, len(docs)))
    return ranges


def handle_transpositions(
    sim_matrix: np.ndarray, n_features: int, n_trans: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Handle multiple transpositions per image."""
    n_images = n_features // n_trans
    assert n_features % n_trans == 0, "Features must be divisible by transpositions"

    # Reshape to get all transposition combinations
    sim_trans = (
        sim_matrix.reshape(n_images, n_trans, n_images, n_trans)
        .transpose(0, 2, 1, 3)
        .reshape(n_images, n_images, n_trans * n_trans)
    )

    # Find best transposition pairs
    best_trans = sim_trans.argmax(axis=2, keepdims=True)
    sim_matrix = np.take_along_axis(sim_trans, best_trans, axis=2).squeeze(axis=2)
    tr_i, tr_j = np.divmod(best_trans.squeeze(axis=2), n_trans)

    return sim_matrix, tr_i, tr_j


class ComputeSimilarity(LoggedTask):
    """
    Compute the similarity between images inside a dataset
    """

    def __init__(
        self, dataset: Dataset, parameters: Optional[dict] = None, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.results: SimilarityResults | dict = {}

        self.dataset = dataset
        self.images = self.dataset.prepare()
        # Sequences of indices for each document to compare
        self.doc_ranges = _make_diff_ranges([i.document for i in self.images])

        self.feat_net = parameters.get("feat_net", FEAT_NET) if parameters else FEAT_NET
        self.topk = parameters.get("topk", COS_TOPK)
        self.algorithm = parameters.get("algorithm", "cosine")

        # Whether to perform pre-filter using cosine similarity to keep only best matches before running segswap
        self.segswap_prefilter = parameters.get("segswap_prefilter", True)
        # If so, how many best matches should be kept
        self.segswap_n = parameters.get("segswap_n", COS_TOPK)

        self.raw_transpositions: List[str] = parameters.get("transpositions", ["none"])
        self.transpositions = [
            getattr(AllTranspose, t.upper()) for t in self.raw_transpositions
        ]

        self.device = get_device()

    @torch.no_grad()
    def get_features(self, img_paths: List[str]):
        """
        Extract features from a list of images
        """
        extractor = FeatureExtractor(
            feat_net=self.feat_net,
            device=self.device,
        )

        img_dataset = FileListDataset(
            data_paths=img_paths,
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

        try:
            del extractor
        except Exception:
            self.print_and_log_warning(
                "[task.similarity] Failed to clear memory from extractor"
            )

        return features

    def format_parameters(self) -> SimParameters:
        return {
            "algorithm": self.algorithm,
            "topk": self.topk,
            "feat_net": self.feat_net,
            "segswap_prefilter": self.segswap_prefilter,
            "segswap_n": self.segswap_n,
            "raw_transpositions": self.raw_transpositions,
            "transpositions": self.transpositions,
        }

    def format_results(
        self, pairs: PairList, doc_uids: Optional[Set[str]] = None
    ) -> SimilarityResults:
        """
        Format the results for output
        TODO change pairs processing

        Args:
            pairs (List[PairTuple]): The similarity pairs
            doc_uids (List[str]): The list of document uids to include in the output

        Returns:
            A dictionary with the document index and pairs
        """
        if doc_uids is None:
            doc_uids = {doc.uid for doc in self.dataset.documents}

        return {
            "parameters": self.format_parameters(),
            "index": {
                "sources": {
                    doc.uid: doc.to_dict()
                    for doc in self.dataset.documents
                    if doc.uid in doc_uids
                },
                "images": [
                    cast(ImageDict, {**im.to_dict(), "doc_uid": im.document.uid})
                    for im in self.images
                ],
                "transpositions": self.raw_transpositions,
            },
            "pairs": [Pair.from_tuple(pair).to_tuple() for pair in pairs],
        }

    @torch.no_grad()
    def compute_similarity(self) -> SimilarityResults:
        """
        Compute the similarity between images in the dataset and returns the results
        """
        source_paths = [str(i.path) for i in self.images]

        self.print_and_log(
            f"[task.similarity] Prepared {len(self.images)} images to be processed"
        )
        topk = self.segswap_n if self.algorithm == "segswap" else self.topk
        features = self.get_features(source_paths)

        # TODO skip this step if self.algorithm == "segswap" && self.segswap_prefilter == false
        pairs = self.compute_cosine_similarity(
            features.cpu().numpy(),
            topk=topk,
            n_transpositions=len(self.transpositions),
        )

        if self.algorithm == "segswap":
            pairs = self.compute_segswap_similarity(
                source_paths, pairs, cos_topk=topk, device=self.device
            )

        all_pairs = list(set().union(*pairs.values()))

        self.print_and_log(
            f"[task.similarity] Computed similarity scores for {len(all_pairs)} pairs"
        )

        return self.format_results(all_pairs)

    def get_top_pairs(
        self,
        sim_matrix: np.ndarray,
        doc1_idx: Iterable[int],
        doc2_idx: Iterable[int],
        topk: Optional[int],
        min_tr_i: np.ndarray,
        min_tr_j: np.ndarray,
    ) -> Set[PairTuple]:
        """Get top-k matches between two documents."""
        submatrix = sim_matrix[doc1_idx][:, doc2_idx]
        tops = np.argsort(-submatrix, axis=1)[
            :, : (topk or self.topk)
        ]  # Negative for descending order

        return {
            (
                min(doc1_idx[i], doc2_idx[j]),
                max(doc1_idx[i], doc2_idx[j]),
                float(submatrix[i, j]),
                int(min_tr_i[i, j]),
                int(min_tr_j[i, j]),
            )
            for i, row in enumerate(tops)
            for j in row
            if doc1_idx[i] != doc2_idx[j]
        }

    @staticmethod
    def get_docs_ref(uid1, uid2) -> DocRef:
        return tuple(sorted([uid1, uid2]))

    def get_doc_uid(self, idx: int) -> str:
        """Get the document UID for an image index."""
        return self.images[idx].document.uid

    def store(self, doc1_uid, doc2_uid, pairs: PairList, algorithm="cosine"):
        """Store similarity pairs for a document pair and sends results to front"""
        doc_ref = self.get_docs_ref(doc1_uid, doc2_uid)

        score_file = SCORES_PATH / self.experiment_id / f"{algorithm}-{doc_ref}.json"
        score_file.parent.mkdir(parents=True, exist_ok=True)

        res = self.format_results(pairs, {doc1_uid, doc2_uid})
        with open(score_file, "wb") as f:
            f.write(orjson.dumps(res, default=serializer))

        if self.algorithm == algorithm:
            file_path = f"{self.experiment_id}/{doc_ref}"
            self.notifier(
                "PROGRESS",
                output={
                    "dataset_url": self.dataset.get_absolute_url(),
                    "annotations": [{doc_ref: get_file_url(DEMO_NAME, file_path)}],
                },
            )

    def compute_cosine_similarity(
        self,
        features: np.ndarray,
        topk: int = COS_TOPK,
        n_transpositions: int = 1,
    ) -> DocPairs:
        """
        Compute pairwise cosine similarities between feature vectors, optionally handling transpositions.

        Args:
            features: Feature vectors of shape (n_samples, n_features)
            topk: Number of most similar matches to return per vector
            n_transpositions: Number of consecutive features representing the same image

        Returns:
            List of tuples (idx1, idx2, similarity, trans1, trans2) where:
            - idx1, idx2: Indices of the matched feature vectors (idx1 < idx2)
            - similarity: Cosine similarity score
            - trans1, trans2: Best matching transposition indices
        """
        doc_ranges = self.doc_ranges or [range(len(features))]

        self.print_and_log(
            f"[task.similarity] Computing cosine similarity for {len(doc_ranges)} pairs"
        )

        sim_matrix = 1.0 - squareform(pdist(features, metric="cosine"))
        n_features = len(features)

        if n_transpositions > 1:
            sim_matrix, tr_i, tr_j = handle_transpositions(
                sim_matrix, n_features, n_transpositions
            )
        else:
            tr_i = tr_j = np.zeros_like(sim_matrix)

        np.fill_diagonal(sim_matrix, -1000)  # Exclude self-matches

        all_pairs: DocPairs = {}
        for doc1_range in doc_ranges:
            doc1_uid = self.get_doc_uid(doc1_range.start)

            for doc2_range in doc_ranges:
                doc2_uid = self.get_doc_uid(doc2_range.start)

                pairs = self.get_top_pairs(
                    sim_matrix, doc1_range, doc2_range, topk, tr_i, tr_j
                )

                doc_refs = self.get_docs_ref(doc1_uid, doc2_uid)
                if doc_refs not in all_pairs:
                    all_pairs[doc_refs] = set()
                all_pairs[doc_refs].update(sorted(pairs))

                self.store(doc1_uid, doc2_uid, pairs, algorithm="cosine")

        return all_pairs

    @torch.no_grad()
    def compute_segswap_similarity(
        self,
        source_images: List[str],
        doc_pairs: DocPairs,
        cos_topk,
        device="cuda",
    ) -> DocPairs:
        """
        Compute the similarity between pairs of images using the SegSwap algorithm

        Args:
            source_images (List[str]): The list of image paths
            doc_pairs (List[tuple[int, int, *any]]): The cosine similarity pairs (i, j, *any)
            cos_topk (int): The number of best matches to return
            device (str): The device to run the computation on

        Returns:
            A list of pairs (k_i, k_j, sim, tr_i, tr_j)
        """
        self.print_and_log(
            f"[task.similarity] Computing SegSwap similarity for {len(list(doc_pairs.keys()))} documents"
        )

        param = torch.load(get_model_path("hard_mining_neg5"), map_location=device)
        backbone = segswap.load_backbone(param).to(device)
        encoder = segswap.load_encoder(param).to(device)

        feat_size = MAX_SIZE // SEG_STRIDE
        y_grid, x_grid = np.where(np.ones((feat_size, feat_size), dtype=bool))

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

        last_img_idx = None
        segswap_scores = {}
        # TODO make a dataloader
        for (doc1_uid, doc2_uid), pairs in doc_pairs.items():
            q_tensor = None
            doc_scores: PairList = []

            batched_pairs = [
                pairs[i : i + cos_topk] for i in range(0, len(pairs), cos_topk)
            ]
            for batch in batched_pairs:
                tensor1, tensor2, batch_pairs = [], [], []
                for i, j, *_ in batch:
                    # Reuse tensor if same image index (assumes sorted pairs)
                    if last_img_idx != i:
                        q_tensor = img_dataset[i]
                        last_img_idx = i

                    tensor1.append(q_tensor)
                    tensor2.append(img_dataset[j])
                    batch_pairs.append((i, j))

                scores = segswap.compute_score(
                    torch.stack(tensor1),
                    torch.stack(tensor2),
                    backbone,
                    encoder,
                    y_grid,
                    x_grid,
                )
                doc_scores.extend(
                    [
                        (i, j, float(s), 0, 0)
                        for (i, j, *_), s in zip(batch_pairs, scores)
                    ]
                )
            self.store(doc1_uid, doc2_uid, doc_scores, "segswap")
            segswap_scores[self.get_docs_ref(doc1_uid, doc2_uid)] = doc_scores

        return segswap_scores

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
            # TODO change to use results_url
            return {
                "dataset_url": self.dataset.get_absolute_url(),
                "annotations": scores,
            }

        try:
            self.results = self.compute_similarity()

            (SCORES_PATH / self.experiment_id).parent.mkdir(parents=True, exist_ok=True)
            with open(
                SCORES_PATH / self.experiment_id / f"{self.dataset.uid}-scores.json",
                "wb",
            ) as f:
                f.write(orjson.dumps(self.results, default=serializer))

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
        # TODO check for scores using same algorithm
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
