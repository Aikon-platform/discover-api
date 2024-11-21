import dramatiq
from typing import Optional

from .const import SIM_QUEUE
from .lib.similarity import ComputeSimilarity
from ..shared.utils.logging import notifying, TLogger, LoggerHelper
from ..shared.dataset import Dataset


# @notifying TODO implement results return with notifying
@dramatiq.actor(time_limit=1000 * 60 * 60, max_retries=0, store_results=True, queue_name=SIM_QUEUE)
def compute_similarity(
    experiment_id: str,
    dataset_uid: str,
    parameters: Optional[dict] = None,
    notify_url: Optional[str] = None,
    tracking_url: Optional[str] = None,
    logger: TLogger = LoggerHelper,
    **kwargs
):
    """
    Run similarity retrieval task on all pairs of documents

    Args:
    - experiment_id: the ID of the similarity retrieval task
    - dataset_uid: the ID of the dataset to use
    - parameters: a JSON object containing the task parameters (model)
    - notify_url: the URL to be called when the task is finished
    - logger: a logger object
    """

    dataset = Dataset(uid=dataset_uid, load=True)

    similarity_task = ComputeSimilarity(
        logger,
        experiment_id=experiment_id,
        dataset=dataset,
        parameters=parameters,
        notify_url=notify_url,
        tracking_url=tracking_url
    )
    similarity_task.run_task()
