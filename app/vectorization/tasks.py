import dramatiq
from typing import Optional
from datetime import datetime, timedelta
import os
import shutil

from ..config import TIME_LIMIT
from .const import VEC_QUEUE  # , IMG_PATH
from .vectorization import LoggedComputeVectorization
from ..shared.utils.logging import notifying, TLogger, LoggerHelper


@dramatiq.actor(
    time_limit=TIME_LIMIT, max_retries=0, store_results=True, queue_name=VEC_QUEUE
)
# @notifying TODO implement results return with notifying
def compute_vectorization(
    experiment_id: str,
    documents: dict,
    model: str,
    notify_url: Optional[str] = None,
    tracking_url: Optional[str] = None,
    logger: TLogger = LoggerHelper,
):
    """
    Run vecto task on list of URL

    Parameters:
    - experiment_id: the ID of the vecto task
    - dataset: dictionary containing the documents to be vectorized
    - notify_url: the URL to be called when the task is finished
    - logger: a logger object
    - doc_id : the id of the annotated witness

    E.g. of dataset dict
    {
    "wit4_man19_0023_260,1335,1072,1114": "http://localhost:8182/iiif/2/wit4_man19_0023.jpg/260,1335,1072,1114/full/0/default.jpg",
    "wit4_man19_0025_244,1462,768,779": "http://localhost:8182/iiif/2/wit4_man19_0025.jpg/244,1462,768,779/full/0/default.jpg",
    "wit4_man19_0030_15,1523,623,652": "http://localhost:8182/iiif/2/wit4_man19_0030.jpg/15,1523,623,652/full/0/default.jpg"
    }
    """
    vectorization_task = LoggedComputeVectorization(
        logger,
        experiment_id=experiment_id,
        documents=documents,
        model=model,
        notify_url=notify_url,
        tracking_url=tracking_url,
    )
    vectorization_task.run_task()


@dramatiq.actor
def delete_images():
    # Function to delete images after a week
    week_ago = datetime.now() - timedelta(days=7)

    # TODO delete images associated with a vectorization

    # for vec_dir in os.listdir(IMG_PATH):
    #     dir_path = os.path.join(IMG_PATH, vec_dir)
    #     if os.path.isdir(dir_path):
    #         dir_modified_time = datetime.fromtimestamp(os.path.getmtime(dir_path))
    #         if dir_modified_time < week_ago:
    #             shutil.rmtree(dir_path, ignore_errors=False, onerror=None)
