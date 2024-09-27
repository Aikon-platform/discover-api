from flask import request, send_from_directory, Blueprint, jsonify
from slugify import slugify
import uuid

from .tasks import compute_vectorization
from ..shared import routes as shared_routes
from ..shared.utils.fileutils import delete_directory
from .const import VEC_RESULTS_PATH, IMG_PATH

from ..shared.utils.logging import console

blueprint = Blueprint("vectorization", __name__, url_prefix="/vectorization")


@blueprint.route("start", methods=["POST"])
@shared_routes.get_client_id
@shared_routes.error_wrapper
def start_vectorization(client_id):
    """
    TODO update that to fit what is sent by the frontend
    {
        "doc_id": "wit17_img17_anno17"
        "model": "0036"
        "callback": "https://domain-name.com/receive-vecto",
        "tracking_url": "url for updates",
        "images": {
            "img_name": "https://domain-name.com/image_name.jpg",
            "img_name": "https://other-domain.com/image_name.jpg",
            "img_name": "https://iiif-server.com/.../coordinates/size/rotation/default.jpg",
            "img_name": "..."
        }
    }
    A list of images to download + information
    """
    if not request.is_json:
        return "No JSON in request: Vectorization task aborted!"

    json_param = request.get_json()

    experiment_id = json_param.get("experiment_id")
    documents = json_param.get("documents", {})
    model = json_param.get("model", None)
    notify_url = json_param.get("callback", None)
    tracking_url = json_param.get("tracking_url")

    return shared_routes.start_task(
        compute_vectorization,
        experiment_id,
        {
            "documents": documents,
            "model": model,
            "notify_url": notify_url,
            "tracking_url": tracking_url,
        },
    )


@blueprint.route("<tracking_id>/cancel", methods=["POST"])
def cancel_vectorization(tracking_id: str):
    return shared_routes.cancel_task(tracking_id)


@blueprint.route("<tracking_id>/status", methods=["GET"])
def status_vectorization(tracking_id: str):
    return shared_routes.status(tracking_id, compute_vectorization)


@blueprint.route("qsizes", methods=["GET"])
def qsizes_vectorization():
    """
    List the queues of the broker and the number of tasks in each queue
    """
    return shared_routes.qsizes(compute_vectorization.broker)


@blueprint.route("monitor", methods=["GET"])
def monitor_vectorization():
    return shared_routes.monitor(VEC_RESULTS_PATH, compute_vectorization.broker)


@blueprint.route("delete_and_relaunch", methods=["POST"])
@shared_routes.get_client_id
@shared_routes.error_wrapper
def delete_and_relaunch(client_id):
    if not request.is_json:
        return "No JSON in request: Vectorization task aborted!"

    data = request.get_json()
    experiment_id = slugify(request.form.get("experiment_id", str(uuid.uuid4())))
    # dict of document ids with a URL containing a list of images
    dataset = data.get("images", {})
    # which url to send back the vectorization results and updates on the task
    notify_url = data.get("callback", None)
    doc_id = data.get("doc_id", None)
    model = data.get("model", None)

    cleared_img_dir = delete_directory(f"{IMG_PATH}/{doc_id}")

    if cleared_img_dir:
        start_response = shared_routes.start_task(
            compute_vectorization,
            experiment_id,
            {
                "dataset": dataset,
                "notify_url": notify_url,
                "doc_id": doc_id,
                "model": model,
            },
        )
        return jsonify({"cleared_img_dir": 1, "start_vectorization": start_response})
    else:
        return jsonify(
            {
                "cleared_img_dir": 0,
                "start_vectorization": "Directory deletion failed, vectorization not started.",
            }
        )


# TODO add clear_doc + clear_old_vectorization routes (see similarity.routes)
