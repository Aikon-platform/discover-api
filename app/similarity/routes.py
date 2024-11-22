import os

from flask import request, Blueprint, jsonify
from slugify import slugify

from .tasks import compute_similarity
from ..shared import routes as shared_routes
from ..shared.utils.fileutils import clear_dir
from .const import (
    IMG_PATH,
    FEATS_PATH,
    SIM_RESULTS_PATH,
    SIM_XACCEL_PREFIX, MODEL_PATH,
)

from .lib.const import FEAT_NET, FEAT_SET, FEAT_LAYER
from ..shared.utils.logging import console

blueprint = Blueprint("similarity", __name__, url_prefix="/similarity")


@blueprint.route("start", methods=["POST"])
@shared_routes.get_client_id
@shared_routes.error_wrapper
def start_similarity(client_id):
    """
    Compute similarity for images from a list of URLs.

    TODO use shared_routes.receive_task

    .. code-block:: python

        documents = {
            "wit3_man186_anno181": "https://eida.obspm.fr/eida/wit3_man186_anno181/list/",
            "wit87_img87_anno87": "https://eida.obspm.fr/eida/wit87_img87_anno87/list/",
            "wit2_img2_anno2": "https://eida.obspm.fr/eida/wit2_img2_anno2/list/"
        }
    
    Each URL corresponds to a document and contains a list of images to download:

    .. code-block:: json

        {
            "img_name": "https://domain-name.com/image_name.jpg",
            "img_name": "https://other-domain.com/image_name.jpg",
            "img_name": "https://iiif-server.com/.../coordinates/size/rotation/default.jpg",
            "img_name": "..."
        }
    
    Each document is compared to itself and other documents resulting in a list a comparison pairs
    """

    if not request.is_json:
        return "No JSON in request: Similarity task aborted!"

    json_param = request.get_json()
    console(json_param, color="cyan")

    experiment_id = json_param.get("experiment_id")
    # dict of document ids with a URL containing a list of images
    dataset = json_param.get("documents", {})
    parameters = {
        # which feature extraction backbone to use
        "feat_net": json_param.get("model", FEAT_NET),
        "feat_set": json_param.get("feat_set", FEAT_SET),
        "feat_layer": json_param.get("feat_layer", FEAT_LAYER),
        "client_id": client_id,
    }
    # which url to send back the similarity results and updates on the task
    notify_url = json_param.get('notify_url', None) or json_param.get('callback', None)
    tracking_url = json_param.get("tracking_url")

    return shared_routes.start_task(
        compute_similarity,
        experiment_id,
        {
            "dataset": dataset,
            "parameters": parameters,
            "notify_url": notify_url,
            "tracking_url": tracking_url,
        },
    )


@blueprint.route("<tracking_id>/cancel", methods=["POST"])
def cancel_similarity(tracking_id: str):
    return shared_routes.cancel_task(tracking_id)


@blueprint.route("<tracking_id>/status", methods=["GET"])
def status_similarity(tracking_id: str):
    return shared_routes.status(tracking_id, compute_similarity)


@blueprint.route("task/<doc_pair>/result", methods=["GET"])
def result_similarity(doc_pair: str):
    """
    Sends the similarity results file for a given document pair
    """
    return shared_routes.result(
        SIM_RESULTS_PATH, SIM_XACCEL_PREFIX, f"{slugify(doc_pair)}.npy"
    )


@blueprint.route("qsizes", methods=["GET"])
def qsizes_similarity():
    """
    List the queues of the broker and the number of tasks in each queue
    """
    return shared_routes.qsizes(compute_similarity.broker)


@blueprint.route("monitor", methods=["GET"])
def monitor_similarity():
    return shared_routes.monitor(SIM_RESULTS_PATH, compute_similarity.broker)


@blueprint.route("monitor/clear/", methods=["POST"])
def clear_old_similarity():
    return {
        "cleared_img_dir": clear_dir(IMG_PATH),
        "cleared features": clear_dir(FEATS_PATH, path_to_clear="*.pt"),
        "cleared_results": clear_dir(SIM_RESULTS_PATH, path_to_clear="*.npy"),
    }


@blueprint.route("monitor/clear/<doc_id>/", methods=["POST"])
def clear_doc(doc_id: str):
    """
    Clear all images, features and scores related to a given document
    """

    # doc_id = "{doc_id}"
    # TODO: re-united doc_id / tracking_id

    return {
        "cleared_img_dir": clear_dir(
            IMG_PATH, path_to_clear=f"*{doc_id}*", force_deletion=True
        ),
        "cleared features": clear_dir(
            FEATS_PATH, path_to_clear=f"*{doc_id}*.pt", force_deletion=True
        ),
        "cleared_results": clear_dir(
            SIM_RESULTS_PATH, path_to_clear=f"*{doc_id}*.npy", force_deletion=True
        ),
    }


@blueprint.route("models", methods=['GET'])
def get_models():
    # TODO make it dynamic with the models in the folder
    models_info = {
        "vit": {
            "name": "Vision Transformer",
            "model": "dino_vitbase8_pretrain",
            "desc": "A transformer model for image classification.",
        },
        "resnet": {
            "name": "ResNet 34",
            "model": "resnet34",
            "desc": "A deep residual network for image classification.",
        },
        "moco": {
            "name": "MoCo v2 800ep",
            "model": "moco_v2_800ep_pretrain",
            "desc": "A contrastive learning model for image classification.",
        },
        "dino": {
            "name": "DINO DeiT-Small 16",
            "model": "dino_deitsmall16_pretrain",
            "desc": "A Vision Transformer model for image classification.",
        },
    }

    try:
        return jsonify(models_info)
    except Exception:
        return jsonify("No models.")

