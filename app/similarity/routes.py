"""
Routes for the similarity API.

Routes:

- POST ``/similarity/start``:
    Starts the similarity process for a dataset.

    - Parameters: see start_similarity
    - Response: JSON object containing the task ID and experiment ID.

- POST ``/similarity/<tracking_id>/cancel``:
    Cancel a similarity task.

    - Parameters:
        - ``tracking_id``: The task ID.
    - Response: JSON object indicating the cancellation status.

- GET ``/similarity/<tracking_id>/status``:
    Get the status of a similarity task.

    - Response: JSON object containing the status of the task.

- GET ``/similarity/qsizes``:
    List the queues of the broker and the number of tasks in each queue.

    - Response: JSON object containing the queue sizes.

- GET ``/similarity/monitor``:
    Monitor the tasks of the broker.

    - Response: JSON object containing the monitoring information.

- GET ``/similarity/models``:
    Get the list of available models.

    - Response: JSON object containing the models and their modification dates.

- POST ``/similarity/clear``:
    Clear the images of a dataset.

    - Parameters:
        - ``dataset_id``: The ID of the dataset.
    - Response: JSON object indicating the number of cleared image directories.


"""

from flask import request, Blueprint, jsonify
from slugify import slugify
import os, time

from .tasks import compute_similarity
from ..shared import routes as shared_routes
from ..shared.utils.fileutils import clear_dir
from .const import (
    # IMG_PATH,
    # FEATS_PATH,
    SIM_RESULTS_PATH,
    SIM_XACCEL_PREFIX,
    MODEL_PATH,
)

from .lib.const import FEAT_NET
from .lib.models import list_known_models
from ..shared.utils.logging import console

blueprint = Blueprint("similarity", __name__, url_prefix="/similarity")


@blueprint.route("start", methods=["POST"])
@shared_routes.get_client_id
@shared_routes.error_wrapper
def start_similarity(client_id):
    """
    Compute similarity for a dataset of images.

    Expected request format:

    .. code-block:: json

        {
            ...(tasking.routes.receive_task request)...
            "parameters": {
                "algorithm": "algorithm",
                "feat_net": "model.pt",
                "feat_set": "set",
                "feat_layer": "layer",
                "segswap_prefilter": true, # if algorithm is "segswap"
                "segswap_n": 0, # if algorithm is "segswap"
            }
        }
    """

    if not request.is_json:
        return "No JSON in request: Similarity task aborted!"

    (
        experiment_id,
        notify_url,
        tracking_url,
        dataset,
        param,
    ) = shared_routes.receive_task(request)

    parameters = {
        "feat_net": param.get("feat_net", FEAT_NET),
        "algorithm": param.get("algorithm", "cosine"),
        "cosine_n_filter": param.get("cosine_n_filter", 10),
        "segswap_prefilter": param.get("segswap_prefilter", True),
        "segswap_n": param.get("segswap_n", 10),
        "client_id": client_id,
    }

    return shared_routes.start_task(
        compute_similarity,
        experiment_id,
        {
            "dataset_uid": dataset.uid,
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
    # TODO clear images and features associated with an old similarity
    return {
        # "cleared_img_dir": clear_dir(IMG_PATH),
        # "cleared features": clear_dir(FEATS_PATH, path_to_clear="*.pt"),
        "cleared_results": clear_dir(SIM_RESULTS_PATH, path_to_clear="*.npy"),
    }


@blueprint.route("monitor/clear/<doc_id>/", methods=["POST"])
def clear_doc(doc_id: str):
    """
    Clear all images, features and scores related to a given document
    """

    # doc_id = "{doc_id}"
    # TODO: delete images and features associated with similarity

    return {
        # "cleared_img_dir": clear_dir(
        #     IMG_PATH, path_to_clear=f"*{doc_id}*", force_deletion=True
        # ),
        # "cleared features": clear_dir(
        #     FEATS_PATH, path_to_clear=f"*{doc_id}*.pt", force_deletion=True
        # ),
        "cleared_results": clear_dir(
            SIM_RESULTS_PATH, path_to_clear=f"*{doc_id}*.npy", force_deletion=True
        ),
    }


@blueprint.route("models", methods=["GET"])
def get_models():
    models_info = list_known_models()

    try:
        return jsonify(models_info)
    except Exception:
        return jsonify("No models.")
