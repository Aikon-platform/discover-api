import json
import os
import time

from flask import request, jsonify, Blueprint

from ..main import app
from .tasks import extract_objects
from ..shared import routes as shared_routes
from .const import ANNO_PATH, MODEL_PATH, IMG_PATH, EXT_XACCEL_PREFIX
from ..shared.utils.fileutils import delete_path, sanitize_str
from ..shared.utils.logging import console
from ..shared.dataset import Dataset

blueprint = Blueprint("regions", __name__, url_prefix="/regions")


@blueprint.route("start", methods=["POST"])
@shared_routes.get_client_id
@shared_routes.error_wrapper
def start_regions_extraction(client_id):
    """
    manifests = {
        "wit3": "https://eida.obspm.fr/eida/iiif/auto/wit3_man186_anno181/manifest.json",
        "wit87": "https://eida.obspm.fr/eida/iiif/auto/wit87_img87_anno87/manifest.json",
        "wit2": "https://eida.obspm.fr/eida/iiif/auto/wit2_img2_anno2/manifest.json"
    }
    Extract regions for images from a list of IIIF URLs.
    """

    # if request.is_json:
    #     json_param = request.get_json()
    # else:
    #     json_param = request.form.to_dict()
    #
    # if not json_param:
    #     return "No data in request: Regions extraction task aborted!"
    #
    # console(json_param, color="cyan")
    # experiment_id = json_param.get('experiment_id')
    # notify_url = json_param.get('notify_url') or json_param.get('callback')
    # tracking_url = json_param.get("tracking_url")

    experiment_id, notify_url, tracking_url, param = shared_routes.receive_task(
        request, ["dataset", "documents", "model"]
    )

    dataset = param.get('dataset')

    documents = param.get('documents', {})
    if type(documents) is str:
        documents = json.loads(documents)

    dataset = Dataset(dataset, documents=documents)
    dataset.save()

    model = param.get('model')

    return shared_routes.start_task(
        extract_objects,
        experiment_id,
        {
            "dataset": dataset,
            "model": model,
            "notify_url": notify_url,
            "tracking_url": tracking_url,
        },
    )


@blueprint.route("<tracking_id>/cancel", methods=["POST"])
def cancel_regions_extraction(tracking_id: str):
    return shared_routes.cancel_task(tracking_id)


@blueprint.route("<tracking_id>/status", methods=["GET"])
def status_regions_extraction(tracking_id: str):
    return shared_routes.status(tracking_id, extract_objects)


@blueprint.route("qsizes", methods=["GET"])
def qsizes_regions_extraction():
    """
    List the queues of the broker and the number of tasks in each queue
    """
    return shared_routes.qsizes(extract_objects.broker)


@blueprint.route("monitor", methods=["GET"])
def monitor_regions_extraction():
    return shared_routes.monitor(ANNO_PATH, extract_objects.broker)


# @blueprint.route("<tracking_id>/result", methods=["GET"])
# def result_extraction(tracking_id: str):
#     return shared_routes.result(tracking_id, ANNO_PATH, EXT_XACCEL_PREFIX, "json")


@blueprint.route("models", methods=['GET'])
def get_models():
    models_info = {}

    try:
        for filename in os.listdir(MODEL_PATH):
            if filename.endswith(".pt"):
                full_path = os.path.join(MODEL_PATH, filename)
                modification_date = os.path.getmtime(full_path)
                models_info[filename] = time.ctime(modification_date)

        return jsonify(models_info)

    except Exception:
        return jsonify("No models.")


@blueprint.route("clear", methods=["POST"])
def clear_images():
    dataset_id = request.form['dataset_id']
    # TODO change inside front ends to send correct path for dataset
    #  (for AIKON: "{app_name}_{digit_ref}" / for dicover-demo "{dataset_id}")
    return {
        "cleared_img_dir": 1 if delete_path(IMG_PATH / dataset_id) else 0,
    }
