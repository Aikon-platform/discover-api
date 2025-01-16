"""
Util functions for tasks
"""

import dramatiq
from dramatiq.middleware import CurrentMessage
from typing import Optional, Callable, Dict, Any, List

from .const import DEMO_NAME
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

        self.error_list: List[str] = []

    def task_update(self, event: str, message: Optional[Any] = None) -> None:
        if self.notifier:
            if event == "ERROR":
                if message and isinstance(message, list):
                    msg = ", ".join(message)
                else:
                    msg = message or "Unknown error"
                self.notifier(event, message=msg)
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
    notify_url: Optional[str] = None,
    logger: TLogger = LoggerHelper,
    notifier=None,
    **task_kwargs,  # Replace with extra parameters needed for the task
):
    """
    Template for a task (see the source code)
    """
    current_task = CurrentMessage.get_current_message()
    current_task_id = current_task.message_id

    task_instance = LoggedTask(
        logger=logger,
        experiment_id=experiment_id,
        notify_url=notify_url,
        notifier=notifier,
        **task_kwargs,  # Replace with extra parameters needed for the task
    )
    task_instance.run_task()

    # json to be dispatch to frontend with @notifying, triggering SUCCESS event
    return {
        "result_url": f"{BASE_URL}/{DEMO_NAME}/{current_task_id}/result",
    }
