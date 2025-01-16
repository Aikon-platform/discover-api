import os

from pathlib import Path
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from .const import FEAT_NET
from .models import load_model, get_transforms_for_model
from ...shared.utils.logging import console


class FeatureExtractor:
    def __init__(self, model_path=None, feat_net=FEAT_NET, device="cpu"):
        """
        Load a pre-trained model for features extraction
        # TODO ADD CLIP
        """

        self.feat_net = feat_net
        self.model_path = model_path

        self.extractor_label = f"{self.feat_net}"
        self.device = device
        self.model = None
        self.transforms = get_transforms_for_model(self.feat_net)

    def initialize(self):
        if self.model is not None:
            return
        self.model, self.feat_func = load_model(
            self.model_path, self.feat_net, self.device
        )

    @torch.no_grad()
    def _calc_feats(self, batch):
        return self.feat_func(self.model(batch))

    @torch.no_grad()
    def extract_features(
        self, data_loader, cache_dir: Path = None, cache_id: str = None
    ) -> torch.Tensor:
        """ """
        torch.cuda.empty_cache()
        if cache_dir is not None:
            feat_path = (
                cache_dir / f"{cache_id}_{self.extractor_label}.pt"
            )  # _{self.feat_layer}.pt"

            if os.path.exists(feat_path):
                feats = torch.load(feat_path, map_location=self.device)
                if feats.numel() != 0:
                    console(f"Loaded extracted features from {feat_path}")
                    return feats
                console(
                    f"[extract_features] No cache: recomputing...",
                    color="yellow",
                )

        console(f"[extract_features] Extracting features...")

        self.initialize()
        features = []
        for i, img in enumerate(data_loader):
            features.append(self._calc_feats(img).detach().cpu())

        features = torch.cat(features).to(torch.float16)

        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            torch.save(features, feat_path)

        return features


def scale_feats(features, n_components):
    # UNUSED ???
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)

    if n_components >= 1:
        pca = PCA(n_components=int(n_components), whiten=True, random_state=0)
    elif n_components > 0:
        pca = PCA(n_components=n_components, whiten=True, random_state=0)
    else:
        pca = PCA(n_components=None, whiten=True, random_state=0)
    return pca.fit_transform(features)
