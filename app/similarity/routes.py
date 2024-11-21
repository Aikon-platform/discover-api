"""
Routes for the similarity API.

Routes:

- POST ``/similarity/start``:
    Starts the similarity process for a dataset.
    
    - Parameters:
        - ``experiment_id``: The ID of the experiment.
        - ``notify_url``: The URL to notify when the task is done.
        - ``tracking_url``: The URL to track the task.
        - ``dataset``: The dataset UID to process.
        - ``documents``: The documents to put into the dataset.
        - ``model``: The model to use for the extraction.
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

from flask import request, Blueprint
from slugify import slugify

from .tasks import compute_similarity
from ..shared import routes as shared_routes
from ..shared.utils.fileutils import clear_dir
from .const import (
    IMG_PATH,
    FEATS_PATH,
    SIM_RESULTS_PATH,
    SIM_XACCEL_PREFIX,
)

from .lib.const import FEAT_NET, FEAT_SET, FEAT_LAYER
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
            "crops": [ # optional
                {
                    "document": "wit3_man186_anno181",
                    "source": "imagename.jpg",
                    "crop_id": "crop_id",
                    "relative": [x, y, w, h]
                }, ...
            ],
            "algorithm": "algorithm",
            "feat_net": "model.pt",
            "feat_set": "set",
            "feat_layer": "layer",
            "segswap_prefilter": true, # if algorithm is "segswap"
            "segswap_n": 0, # if algorithm is "segswap"
        }
    """

    if not request.is_json:
        return "No JSON in request: Similarity task aborted!"

    experiment_id, notify_url, tracking_url, dataset, param = shared_routes.receive_task(request)

    parameters = {
        # which feature extraction backbone to use
        "feat_net": param.get("model", FEAT_NET),
        "feat_set": param.get("feat_set", FEAT_SET),
        "feat_layer": param.get("feat_layer", FEAT_LAYER),
        "client_id": client_id,
    }

    return shared_routes.start_task(
        compute_similarity,
        experiment_id,
        {
            "dataset": dataset.uid,
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
