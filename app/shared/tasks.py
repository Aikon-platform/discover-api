import dramatiq
from dramatiq.middleware import CurrentMessage
from typing import Optional, Callable, Dict, Any, List

from .const import DEMO_NAME
from .. import config
from ..shared.utils.logging import notifying, TLogger, LoggerHelper, send_update, LoggingTaskMixin


class Task:
    def __init__(
        self,
        experiment_id: str,
        notify_url: Optional[str] = None,
        tracking_url: Optional[str] = None,
        notifier: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ):
        self.experiment_id = experiment_id
        self.notify_url = notify_url
        self.tracking_url = tracking_url
        self.notifier = notifier

    def run_task(self):
        pass

    def task_update(self, event, message=None):
        if self.tracking_url:
            send_update(self.experiment_id, self.tracking_url, event, message)
        if self.notifier:
            if event == "ERROR":
                self.notifier(event, message=message)
                raise Exception(message)
                # make frontend/task stop

    class Meta:
        abstract = True


class LoggedTask(LoggingTaskMixin, Task):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.error_list: List[str] = []

    def handle_error(self, message: str, exception: Optional[Exception] = None) -> None:
        self.print_and_log_error(f"[task.regions] {message}", e=exception)
        self.error_list.append(f"[API ERROR] {message}")

    def run_task(self):
        pass

    class Meta:
        abstract = True


################################################################
# ⚠️  This is only a template of a task for a given module  ⚠️ #
# ⚠️    This is not meant to be executed, but should be     ⚠️ #
# ⚠️ copied and adapted to the specific needs of the module ⚠️ #
################################################################

@dramatiq.actor(
    time_limit=1000 * 60 * 60,
    max_retries=0,
    queue_name="queue_nb",
    store_results=True
)
@notifying
def abstract_task(
    experiment_id: str,
    notify_url: Optional[str] = None,
    tracking_url: Optional[str] = None,
    logger: TLogger = LoggerHelper,
    notifier=None,
):
    current_task = CurrentMessage.get_current_message()
    current_task_id = current_task.message_id

    task_instance = LoggedTask(
        logger,
        experiment_id=experiment_id,
        notify_url=notify_url,
        tracking_url=tracking_url,
        notifier=notifier,
    )
    task_instance.run_task()

    # json to be dispatch to frontend with @notifying
    return {
        "result_url": f"{config.BASE_URL}/{DEMO_NAME}/{current_task_id}/result"
    }
