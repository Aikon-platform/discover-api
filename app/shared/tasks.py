"""
Util functions for tasks
"""
import uuid

import dramatiq
from dramatiq.middleware import CurrentMessage
from typing import Optional, Callable, Dict, Any, List

from .const import DEMO_NAME
from .dataset import Dataset
from ..config import TIME_LIMIT, BASE_URL
from ..shared.utils.logging import (
    notifying,
    TLogger,
    LoggerHelper,
    LoggingTaskMixin,
)


class LoggedTask(LoggingTaskMixin):
    """
    Base class for tasks that need to log their progress and errors
    - experiment_id: the ID of the vecto task
    - notify_url: the URL to be called when the task is finished

    added by @notifying decorator
    - logger: a logger object
    - notifier: notify() function that can be used to send event updates to the frontend
    """

    def __init__(
        self,
        logger: TLogger,
        experiment_id: str,
        notify_url: Optional[str] = None,
        notifier: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(logger, *args, **kwargs)
        self.experiment_id = experiment_id
        # Frontend endpoint to sends results and task events to
        self.notify_url = notify_url
        # Discover-demo frontend uses /status endpoint to retrieve task message (@dramatiq.store_results=True)

        # notify() function to send any event to frontend
        self.notifier = notifier

        current_task = CurrentMessage.get_current_message()
        self.task_id = current_task.message_id

        self.error_list: List[str] = []

    def task_update(self, event: str, message: Optional[Any] = None, **kwargs) -> None:
        if not self.notifier:
            return
        msg = (
            ", ".join(message)
            if isinstance(message, list)
            else message or self.error_list or "Unknown error"
        )
        self.notifier(event, message=msg, **kwargs)
        if event == "ERROR":
            raise Exception(f"Task {self.experiment_id} failed with error:\n{msg}")

    def handle_error(self, message: str, exception: Optional[Exception] = None) -> None:
        self.print_and_log_error(
            f"[task.{self.__class__.__name__}] {message}", e=exception
        )
        self.error_list.append(f"[API ERROR] {message}")

    def run_task(self) -> bool:
        raise NotImplementedError("Subclasses must implement this method")

    class Meta:
        abstract = True


################################################################
# ⚠️  This is only a template of a task for a given module  ⚠️ #
# ⚠️    This is not meant to be executed, but should be     ⚠️ #
# ⚠️ copied and adapted to the specific needs of the module ⚠️ #
################################################################


@dramatiq.actor(
    time_limit=TIME_LIMIT, max_retries=0, queue_name="queue_nb", store_results=True
)
@notifying
def abstract_task(
    experiment_id: str,
    dataset_uid: str,
    notify_url: Optional[str] = None,
    logger: TLogger = LoggerHelper,
    notifier=None,
    **task_kwargs,  # Replace with extra parameters needed for the task
):
    """
    Template for a task (see the source code)
    """
    dataset = Dataset(dataset_uid, load=True)

    task_instance = LoggedTask(
        experiment_id=experiment_id,  # Used by LoggedTask
        logger=logger,  # Used by LoggedTask
        notify_url=notify_url,  # Used by LoggedTask
        notifier=notifier,  # Used by LoggedTask
        dataset=dataset,
        **task_kwargs,  # Replace with extra parameters needed for the task
    )
    success = task_instance.run_task()

    if success:
        # json to be dispatch to frontend with @notifying, triggering SUCCESS event
        return {doc.uid: doc.get_results_url(DEMO_NAME) for doc in dataset.documents}
    return {"error": task_instance.error_list}
