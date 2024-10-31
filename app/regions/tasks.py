import dramatiq
from dramatiq.middleware import CurrentMessage
from typing import Optional

from .. import config
from .const import EXT_QUEUE, DEMO_NAME
from .lib.regions import LoggedExtractRegions
from ..shared.utils.logging import notifying, TLogger, LoggerHelper


@dramatiq.actor(
    time_limit=1000 * 60 * 60,
    max_retries=0,
    queue_name=EXT_QUEUE,
    store_results=True
)
@notifying
def extract_objects(
    experiment_id: str,
    documents: dict,
    model: Optional[str] = None,
    notify_url: Optional[str] = None,
    tracking_url: Optional[str] = None,
    logger: TLogger = LoggerHelper,
    notifier=None,
):
    current_task = CurrentMessage.get_current_message()
    current_task_id = current_task.message_id

    regions_extraction_task = LoggedExtractRegions(
        logger,
        experiment_id=experiment_id,
        documents=documents,
        model=model,
        notify_url=notify_url,
        tracking_url=tracking_url,
        notifier=notifier,
    )
    regions_extraction_task.run_task()

    # json to be dispatch to frontend with @notifying
    return {"result_url": f"{config.BASE_URL}/{DEMO_NAME}/{current_task_id}/result"}
