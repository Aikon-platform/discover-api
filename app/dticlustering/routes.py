from flask import request, send_from_directory, Blueprint
from slugify import slugify
import uuid
from dramatiq_abort import abort
from dramatiq.results import ResultMissing, ResultFailure
import json
import shutil
from datetime import datetime

from .. import config

from ..main import app
from .tasks import train_dti
from ..shared.utils.fileutils import xaccel_send_from_directory, clear_dir, delete_path
from ..shared import routes as shared_routes
from .const import (
    DTI_RESULTS_PATH,
    DATASETS_PATH,
    RUNS_PATH,
    DTI_DATA_FOLDER,
    DTI_XACCEL_PREFIX,
)


blueprint = Blueprint("clustering", __name__, url_prefix="/clustering")


@blueprint.route("start", methods=["POST"])
def start_clustering():
    """
    Start a new DTI clustering task

    Accepts the following POST parameters:
    - dataset_url [required]: the URL of the zipped dataset to be used
    - experiment_id [optional]: a unique identifier for this clustering task
    - dataset_id [optional]: a unique identifier for the dataset to be used
    - notify_url [optional]: the URL to be called when the task is finished
    - parameters [optional]: a JSON object containing the parameters to be used

    The callback_url will be called with a JSON object containing the following keys:
    - tracking_id: the task ID
    - result_url: the URL from which to fetch the results
    """

    # Extract experiment_id, dataset_id, dataset_url from POST parameters
    dataset_url = request.form["dataset_url"]  # Throw 400 if not exists

    experiment_id = slugify(request.form.get("experiment_id", str(uuid.uuid4())))
    dataset_id = slugify(request.form.get("dataset_id", str(uuid.uuid4())))
    notify_url = request.form.get("notify_url", None)
    parameters = json.loads(request.form.get("parameters", "{}"))

    return shared_routes.start_task(
        train_dti,
        experiment_id,
        {
            "dataset_id": dataset_id,
            "dataset_url": dataset_url,
            "parameters": parameters,
            "notify_url": notify_url,
        },
    )


@blueprint.route("<tracking_id>/cancel", methods=["POST"])
def cancel_clustering(tracking_id: str):
    return shared_routes.cancel_task(tracking_id)


@blueprint.route("<tracking_id>/status", methods=["GET"])
def status_clustering(tracking_id: str):
    return shared_routes.status(tracking_id, train_dti)


@blueprint.route("<tracking_id>/result", methods=["GET"])
def result_clustering(tracking_id: str):
    return shared_routes.result(tracking_id, DTI_RESULTS_PATH, DTI_XACCEL_PREFIX)


@blueprint.route("qsizes", methods=["GET"])
def qsizes_clustering():
    return shared_routes.qsizes(train_dti.broker)


@blueprint.route("monitor", methods=["GET"])
def monitor_clustering():
    return shared_routes.monitor(DTI_DATA_FOLDER, train_dti.broker)


@blueprint.route("monitor/clear", methods=["POST"])
def clear_clustering():
    return {
        "cleared_runs": clear_dir(RUNS_PATH, file_to_check="trainer.log"),
        "cleared_datasets": clear_dir(DATASETS_PATH, file_to_check="ready.meta"),
        "cleared_results": clear_dir(DTI_RESULTS_PATH, path_to_clear="*.zip"),
    }


@blueprint.route("monitor/clear/<tracking_id>/", methods=["POST"])
def clear_run(tracking_id: str):
    return {
        "cleared_runs": 1 if delete_path(RUNS_PATH / tracking_id) else 0,
        "cleared_datasets": 0,
        "cleared_results": 1
        if delete_path(DTI_RESULTS_PATH / f"{tracking_id}.zip")
        else 0,
    }
