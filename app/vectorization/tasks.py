import dramatiq
from typing import Optional
from datetime import datetime, timedelta
import os
import shutil

from ..config import TIME_LIMIT
from .const import VEC_QUEUE  # , IMG_PATH
from .vectorization import ComputeVectorization
from ..shared.utils.logging import notifying, TLogger, LoggerHelper


@dramatiq.actor(
    time_limit=TIME_LIMIT, max_retries=0, store_results=True, queue_name=VEC_QUEUE
)
@notifying
def compute_vectorization(
    documents: dict,
    model: str,
    experiment_id: str,
    notify_url: Optional[str] = None,
    logger: TLogger = LoggerHelper,
    notifier=None,
):
    """
    Run vecto task on list of URL

    Parameters:
    - documents: dictionary containing the documents to be vectorized
    - model: model used to vectorize documents

    E.g. of documents dict
    {
    "wit4_man19_0023_260,1335,1072,1114": "http://localhost:8182/iiif/2/wit4_man19_0023.jpg/260,1335,1072,1114/full/0/default.jpg",
    "wit4_man19_0025_244,1462,768,779": "http://localhost:8182/iiif/2/wit4_man19_0025.jpg/244,1462,768,779/full/0/default.jpg",
    "wit4_man19_0030_15,1523,623,652": "http://localhost:8182/iiif/2/wit4_man19_0030.jpg/15,1523,623,652/full/0/default.jpg"
    }
    """
    vectorization_task = ComputeVectorization(
        documents=documents,
        model=model,
        logger=logger,
        experiment_id=experiment_id,
        notify_url=notify_url,
        notifier=notifier,
    )
    return vectorization_task.run_task()


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
