"""
Region extraction tasks
"""
import dramatiq
from dramatiq.middleware import CurrentMessage
from typing import Optional

from .. import config
from .const import EXT_QUEUE, DEMO_NAME
from .lib.regions import ExtractRegions
from ..shared.utils.logging import notifying, TLogger, LoggerHelper
from ..shared.dataset import Dataset

@dramatiq.actor(
    time_limit=1000 * 60 * 60,
    max_retries=0,
    queue_name=EXT_QUEUE,
    store_results=True
)
@notifying
def extract_objects(
    experiment_id: str,
    dataset: str,
    model: Optional[str] = None,
    notify_url: Optional[str] = None,
    tracking_url: Optional[str] = None,
    logger: TLogger = LoggerHelper,
    notifier=None,
):
    """
    Extract objects from a dataset using a model

    :param experiment_id: the experiment id
    :param dataset: the dataset UID to process
    :param model: the model to use for extraction
    :param notify_url: the URL to notify the frontend
    :param tracking_url: the URL to track the task
    :param logger: the logger to use
    """
    current_task = CurrentMessage.get_current_message()
    current_task_id = current_task.message_id
    dataset = Dataset(dataset, load=True)

    regions_extraction_task = ExtractRegions(
        dataset=dataset,
        model=model,
        logger=logger,
        experiment_id=experiment_id,
        notify_url=notify_url,
        tracking_url=tracking_url,
        notifier=notifier,
    )
    regions_extraction_task.run_task()

    # TODO turn txt result into json

    # json to be dispatch to frontend with @notifying
    return {"result_url": f"{config.BASE_URL}/{DEMO_NAME}/{current_task_id}/result"}
