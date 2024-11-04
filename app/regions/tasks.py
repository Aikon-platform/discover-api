import dramatiq
from typing import Optional

from .const import EXT_QUEUE
from .lib.regions import LoggedExtractRegions
from ..shared.utils.logging import notifying, TLogger, LoggerHelper


@dramatiq.actor(time_limit=1000 * 60 * 60, max_retries=0, queue_name=EXT_QUEUE)
# @notifying
def extract_objects(
    experiment_id: str,
    documents: dict,
    model: Optional[str] = None,
    notify_url: Optional[str] = None,
    tracking_url: Optional[str] = None,
    logger: TLogger = LoggerHelper,
):
    regions_extraction_task = LoggedExtractRegions(
        logger,
        experiment_id=experiment_id,
        documents=documents,
        model=model,
        notify_url=notify_url,
        tracking_url=tracking_url,
    )
    regions_extraction_task.run_task()
