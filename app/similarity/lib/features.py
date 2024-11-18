import os
import sys

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


def load_model(model_path, feat_net, feat_set, device):
    """
    Load a pre-trained model for features extraction
    # TODO ADD CLIP
    """
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


def extract_features(
    data_loader,
    doc_id,
    device,
    feat_net=FEAT_NET,
    feat_set=FEAT_SET,
    feat_layer=FEAT_LAYER,
):
    """
    feat_net ['resnet34', 'moco_v2_800ep_pretrain', 'dino_deitsmall16_pretrain', 'dino_vitbase8_pretrain']
    """
    torch.cuda.empty_cache()
    feat_path = f"{FEATS_PATH}/{doc_id}_{feat_net}_{feat_set}_{feat_layer}.pt"

    with torch.no_grad():
        if os.path.exists(feat_path):
            feats = torch.load(feat_path, map_location=device)
            if feats.numel() != 0:
                console(f"Load extracted features for {doc_id}")
                return feats
            console(
                f"[extract_features] {doc_id} features file is empty: recomputing...",
                color="yellow",
            )
        console(f"[extract_features] Extracting features for {doc_id}...")

        try:
            model_path = get_model_path(feat_net)
            model = load_model(model_path, feat_net, feat_set, device)
        except ValueError as e:
            console("[extract_features] Unable to extract features", e=e)
            return []

        model.eval()
        features = []
        for i, img in enumerate(data_loader):
            features.append(img_feat(img, model, feat_net, feat_layer))

    features = torch.cat(
        features
    )  # .as(torch.float16) #TODO change here to reduce feat size
    torch.save(features, feat_path)

    return features


def img_feat(img, model, feat_net, feat_layer):
    """
    Extract features from a single image
    """
    if "dino" in feat_net:
        feat = model(img).detach().cpu()
    elif feat_layer == "conv4":
        feat = model(img)["conv4"].detach().cpu().flatten(start_dim=1)
    else:
        feat = model(img)["avgpool"].detach().cpu().squeeze()
    return feat


def scale_feats(features, n_components):
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)

    if n_components >= 1:
        pca = PCA(n_components=int(n_components), whiten=True, random_state=0)
    elif n_components > 0:
        pca = PCA(n_components=n_components, whiten=True, random_state=0)
    else:
        pca = PCA(n_components=None, whiten=True, random_state=0)
    return pca.fit_transform(features)
