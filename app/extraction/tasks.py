import dramatiq
from typing import Optional

from .const import EXT_QUEUE
from .lib.extraction import LoggedExtractObjects
from ..shared.utils.logging import notifying, TLogger, LoggerHelper

# @notifying TODO implement results return with notifying
@dramatiq.actor(
    time_limit=1000 * 60 * 60, max_retries=0, store_results=True, queue_name=EXT_QUEUE
)
def extract_objects(
    experiment_id: str,
    manifest_url,
    model: Optional[str] = None,
    notify_url: Optional[str] = None,
    logger: TLogger = LoggerHelper,
):
    extraction_task = LoggedExtractObjects(
        logger, manifest_url=manifest_url, model=model, notify_url=notify_url
    )
    extraction_task.run_task()


# @notifying TODO implement results return with notifying
@dramatiq.actor(time_limit=1000 * 60 * 60, max_retries=0, store_results=True)
def extract_all(
    experiment_id: str,
    manifest_list,
    model: Optional[str] = None,
    notify_url: Optional[str] = None,
    logger: TLogger = LoggerHelper,
):

    for url in manifest_list:
        extraction_task = LoggedExtractObjects(
            logger, manifest_url=url, model=model, notify_url=notify_url
        )
        extraction_task.run_task()
