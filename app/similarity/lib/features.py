import os
import sys

from pathlib import Path
import torch
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision import models
from collections import OrderedDict
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from ..const import FEATS_PATH
from .const import FEAT_LAYER, FEAT_SET, FEAT_NET
from .utils import get_model_path
from .vit import VisionTransformer
from ...shared.utils.logging import console

def _load_model(model_path, feat_net, feat_set, device):
    if model_path is None:
        model_path = get_model_path(feat_net)

    if feat_net == "resnet34" and feat_set == "imagenet":
        model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1).to(
            device
        )
        model = create_feature_extractor(
            model, return_nodes={"layer3.5.bn2": FEAT_LAYER, "avgpool": "avgpool"}
        )

    elif feat_net == "moco_v2_800ep_pretrain" and feat_set == "imagenet":
        model = models.resnet50().to(device)
        pre_dict = torch.load(model_path)["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in pre_dict.items():
            name = k[17:]
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict, strict=False)
        model = create_feature_extractor(
            model, return_nodes={"layer3.5.bn3": FEAT_LAYER, "avgpool": "avgpool"}
        )
    elif feat_net == "dino_deitsmall16_pretrain":
        pre_dict = torch.load(model_path)
        model = VisionTransformer(
            patch_size=16, embed_dim=384, num_heads=6, qkv_bias=True
        ).to(device)
        model.load_state_dict(pre_dict)

    elif feat_net == "dino_vitbase8_pretrain":
        pre_dict = torch.load(model_path)
        model = VisionTransformer(
            patch_size=8, embed_dim=768, num_heads=12, qkv_bias=True
        ).to(device)
        model.load_state_dict(pre_dict)
    else:
        raise ValueError("Invalid network or dataset for feature extraction.")
    
    return model

class FeatureExtractor:
    def __init__(self, model_path=None, feat_net=FEAT_NET, feat_set=FEAT_SET, feat_layer=FEAT_LAYER, device="cpu"):
        """
        Load a pre-trained model for features extraction
        # TODO ADD CLIP
        # TODO make this function more versatile

        feat_net ['resnet34', 'moco_v2_800ep_pretrain', 'dino_deitsmall16_pretrain', 'dino_vitbase8_pretrain']
        feat_set ['imagenet']
        """

        self.feat_net = feat_net
        self.model_path = model_path
        self.feat_set = feat_set
        self.feat_layer = feat_layer
        if "dino" in self.feat_net:
            self.feat_layer = None

        self.extractor_label = f"{self.feat_net}+{self.feat_set}@{self.feat_layer}"
        self.device = device
        self.model = None

    def initialize(self):
        if self.model is not None:
            return
        self.model = _load_model(self.model_path, self.feat_net, self.feat_set, self.device)

    @torch.no_grad()
    def _calc_feats(self, batch):
        if "dino" in self.feat_net:
            return self.model(batch)
        return self.model(batch)[self.feat_layer].flatten(start_dim=1)

    @torch.no_grad()
    def extract_features(self, data_loader, cache_dir: Path=None, cache_id: str=None) -> torch.Tensor:
        """
        """
        torch.cuda.empty_cache()
        if cache_dir is not None:
            feat_path = cache_dir / f"{cache_id}_{self.extractor_label}_{self.feat_layer}.pt"

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

        features = torch.cat(
            features
        ).to(torch.float16)

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
