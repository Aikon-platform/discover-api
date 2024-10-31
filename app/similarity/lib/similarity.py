import os
from typing import Optional
import requests
import numpy as np
import torch
import cv2
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader

from sklearn.metrics.pairwise import cosine_similarity

from ..const import SCORES_PATH, IMG_PATH
from .const import (
    SEG_STRIDE,
    MAX_SIZE,
    COS_TOPK,
    FEAT_NET,
    FEAT_SET,
    FEAT_LAYER,
)
from .dataset import IllusDataset
from .features import extract_features
from .segswap import load_backbone, load_encoder, resize, compute_score

from .utils import get_model_path, doc_pairs

from ...shared.utils import get_device
from ...shared.utils.fileutils import has_content
from ...shared.utils.img import download_images
from ...shared.utils.logging import LoggingTaskMixin, console, send_update


def get_doc_feat(doc_id, feat_net=FEAT_NET, feat_set=FEAT_SET, feat_layer=FEAT_LAYER):
    device = get_device()
    img_dataset = IllusDataset(
        img_dirs=IMG_PATH / doc_id,
        transform=["resize", "normalize"],
        device=device,
    )

    data_loader = DataLoader(img_dataset, batch_size=128, shuffle=False)

    features = (
        extract_features(data_loader, doc_id, device, feat_net, feat_set, feat_layer)
        .cpu()
        .numpy()
    )
    if not len(features) or type(features) is not np.ndarray:
        raise ValueError(f"No feature extracted for {doc_id}")
    return features, img_dataset.get_image_paths()


def doc_sim_pairs(sim_scores, query_doc, sim_doc, is_doc_1=True):
    sim_pairs = []
    tr_ = transforms.Resize((224, 224))
    for i, query_img in enumerate(query_doc):
        # # TODO is it necessary to perform those operations?
        img = cv2.imread(query_img)
        img = torch.from_numpy(img).permute(2, 0, 1)
        tr_img = tr_(img).permute(1, 2, 0).numpy()
        cv2.imwrite(query_img, tr_img)

        query_scores = sim_scores[:][i] if is_doc_1 else sim_scores[i, :]
        # set 0 as similarity score for the query image to itself
        query_scores = [
            0.0 if query_img == sim_img else img_score
            for sim_img, img_score in zip(query_doc, query_scores)
        ]

        top_indices = np.argsort(query_scores)[-COS_TOPK:][::-1]
        sim_pairs.append(
            [
                (query_img, sim_doc[j]) if is_doc_1 else (sim_doc[j], query_img)
                for j in top_indices
            ]
        )

    return sim_pairs


def compute_cos_pairs(doc1, doc2):
    doc1_feat, doc1_imgs = doc1
    doc2_feat, doc2_imgs = doc2

    sim = cosine_similarity(
        doc1_feat, doc2_feat
    )  # sim has shape (n_img_doc1, n_img_doc2)
    sim_pairs = doc_sim_pairs(sim, doc1_imgs, doc2_imgs)
    cos_pairs = np.array(sim_pairs)

    """
    # If we don't assume that all the best matching images of doc2 in doc1
    # are already contained in best matching images of doc1 in doc2
    # (might be the case if an image is never ranked as a COS_TOPK best match)
    sim_pairs += doc_sim_pairs(sim, doc2_imgs, doc1_imgs, False)

    # Remove duplicates pairs with list(set())
    cos_pairs = np.array(list(set(sim_pairs)))
    """

    # cos_pairs = [[(img1doc1, img1doc2), (img1doc1, img2doc2), ...] # COS_TOPK best matching images for img1doc1
    #             [(img2doc1, img4doc2), (img2doc1, img8doc2), ...]] # COS_TOPK best matching images for img2doc1
    return cos_pairs


def segswap_similarity(cos_pairs, output_file=None):
    param = torch.load(get_model_path("hard_mining_neg5"))
    backbone = load_backbone(param)
    encoder = load_encoder(param)

    feat_size = MAX_SIZE // SEG_STRIDE
    mask = np.ones((feat_size, feat_size), dtype=bool)
    y_grid, x_grid = np.where(mask)

    # dtype = [("score", float), ("doc1", "U100"), ("doc2", "U100")]
    # scores_npy = np.empty((0, 3), dtype=dtype)
    scores_npy = np.empty((0, 3), dtype=object)

    norm_mean, norm_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    transformINet = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(norm_mean, norm_std)]
    )

    """
    first_imgs = cos_pairs[:, 0]
    # get a unique list of all the first img in pairs of cosine similar images
    query_imgs = np.unique(first_imgs)
    for q_img in query_imgs:
        img_pairs = cos_pairs(np.char.startswith(first_imgs, q_img))
    """

    # img_pairs = [(img1doc1, img1doc2), (img1doc1, img2doc2), ...] # pairs for img1doc1
    for img_pairs in cos_pairs:
        q_img = os.path.basename(img_pairs[0, 0])

        qt_img = resize(Image.open(img_pairs[0, 0]).convert("RGB"))
        q_tensor = transformINet(qt_img).cuda()
        sim_imgs = img_pairs[:, 1]

        # tensor1 = []
        tensor1 = [q_tensor] * len(sim_imgs)
        tensor2 = []
        for s_img in sim_imgs:  # sim_imgs[:SEG_TOPK]
            st_img = resize(Image.open(s_img).convert("RGB"))
            # tensor1.append(q_tensor)  # NOTE: maybe not necessary to duplicate same img tensor
            tensor2.append(transformINet(st_img).cuda())

        score = compute_score(
            torch.stack(tensor1),
            torch.stack(tensor2),
            backbone,
            encoder,
            y_grid,
            x_grid,
        )

        # q_scores = np.empty(len(score), dtype=dtype)

        for i in range(len(score)):
            s_img = sim_imgs[i]
            pair_score = np.array(
                [[round(score[i], 5), q_img, os.path.basename(s_img)]]
            )
            scores_npy = np.vstack([scores_npy, pair_score])
            # q_scores[i]["score"] = round(score[i], 5)
            # q_scores[i]["doc1"] = q_img
            # q_scores[i]["doc2"] = os.path.basename(sim_imgs[i])

        # scores_npy = np.append(scores_npy, q_scores)

    if output_file:
        try:
            # np.save(output_file, scores_npy, allow_pickle=False)
            np.save(output_file, scores_npy)
        except Exception as e:
            console(f"Failed to save {output_file}.npy", e=e)

    # scores_npy = [(score, img1doc1, img1doc2), # each cosine pair of image is given a score
    #               (score, img1doc1, img2doc2),
    #               (score, img2doc1, img4doc2),
    #               (score, img2doc1, img8doc2),
    #                ...                        ]

    return scores_npy


class ComputeSimilarity:
    def __init__(
        self,
        experiment_id: str,
        dataset: dict,
        parameters: Optional[dict] = None,
        notify_url: Optional[str] = None,
        tracking_url: Optional[str] = None,
    ):
        self.experiment_id = experiment_id
        self.dataset = dataset
        self.notify_url = notify_url
        self.tracking_url = tracking_url

        self.client_id = parameters.get("client_id") if parameters else "default"
        self.feat_net = parameters.get("feat_net", FEAT_NET) if parameters else FEAT_NET
        self.feat_set = parameters.get("feat_set", FEAT_SET) if parameters else FEAT_SET
        self.feat_layer = (
            parameters.get("feat_layer", FEAT_LAYER) if parameters else FEAT_LAYER
        )
        self.doc_ids = []
        self.computed_pairs = []

    def run_task(self):
        pass

    def check_dataset(self):
        # TODO add more checks
        if len(list(self.dataset.keys())) == 0:
            return False
        return True

    def task_update(self, event, message=None):
        if self.tracking_url:
            send_update(self.experiment_id, self.tracking_url, event, message)
            return True
        else:
            return False

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


class LoggedComputeSimilarity(LoggingTaskMixin, ComputeSimilarity):
    def run_task(self):
        if not self.check_dataset():
            self.print_and_log_warning(f"[task.similarity] No documents to compare")
            self.task_update("ERROR", "[API ERROR] No documents to compare")
            return

        self.print_and_log(
            f"[task.similarity] Similarity task triggered for {list(self.dataset.keys())} with {self.feat_net}!"
        )
        self.task_update("STARTED")

        try:
            self.download_documents()
            self.compute_and_send_scores()

            self.print_and_log(
                f"[task.similarity] Successfully send similarity scores for {self.doc_ids}"
            )
            self.task_update("SUCCESS")
            return True

        except Exception as e:
            self.task_update(
                "ERROR",
                f"[API ERROR] Failed to compute and send similarity scores: {e}",
            )

    def download_documents(self):
        # _, _, docs_ids = download_dataset(self.dataset, datasets_dir_path=IMG_PATH, dataset_dir_name=self.client_id)
        # self.doc_ids = docs_ids
        for doc_id, url in self.dataset.items():
            self.print_and_log(
                f"[task.similarity] Processing {doc_id}...", color="blue"
            )
            try:
                doc_id = f"{self.client_id}_{doc_id}"
                self.doc_ids.append(doc_id)
                if not has_content(f"{IMG_PATH}/{doc_id}/"):
                    self.print_and_log(
                        f"[task.similarity] Downloading {doc_id} images..."
                    )
                    download_images(url, doc_id, IMG_PATH, MAX_SIZE)
            except Exception as e:
                self.print_and_log(
                    f"[task.similarity] Unable to download images for {doc_id}", e
                )
                raise Exception

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

    def compute_pairs(self, doc_pair, score_file):
        try:
            doc1 = get_doc_feat(
                doc_pair[0], self.feat_net, self.feat_set, self.feat_layer
            )
            doc2 = get_doc_feat(
                doc_pair[1], self.feat_net, self.feat_set, self.feat_layer
            )
        except Exception as e:
            self.print_and_log(f"Error when extracting features", e=e)
            return False

        try:
            self.print_and_log(f"Computing cosine scores for {doc_pair}")
            cos_pairs = compute_cos_pairs(doc1, doc2)
        except Exception as e:
            self.print_and_log(f"Error when computing cosine similarity", e=e)
            raise Exception
        try:
            self.print_and_log(f"Computing segswap scores for {doc_pair}")
            segswap_similarity(cos_pairs, output_file=score_file)
        except Exception as e:
            self.print_and_log(f"Error when computing segswap scores", e=e)
            raise Exception
        return True
