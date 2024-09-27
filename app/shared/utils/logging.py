import functools
import json
import logging
import os

import time
import dramatiq
from tqdm import tqdm
from dramatiq.middleware import CurrentMessage
from dramatiq.results import Results, ResultBackend
import traceback
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Optional,
    Type,
    TypeVar,
    Union,
)
import requests


"""
A module for logging progress
"""

T = TypeVar("T")


def exc_str(e: Exception):
    return f"[{e.__class__.__name__}] {e}\nStack Trace:\n{traceback.format_exc()}\n"


def pprint(o):
    if isinstance(o, str):
        try:
            return json.dumps(json.loads(o), indent=4, sort_keys=True)
        except ValueError:
            return o
    elif isinstance(o, dict) or isinstance(o, list):
        return json.dumps(o, indent=4, sort_keys=True)
    else:
        return str(o)


class ConsoleColors:
    """
    Last digit
    0	black
    1	red
    2	green
    3	yellow
    4	blue
    5	magenta
    6	cyan
    7	white
    """

    black = "\033[90m"
    red = "\033[91m"
    green = "\033[92m"
    yellow = "\033[93m"
    blue = "\033[94m"
    magenta = "\033[95m"
    cyan = "\033[96m"
    white = "\033[97m"
    bold = "\033[1m"
    underline = "\033[4m"
    end = "\033[0m"


def get_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())


def get_color(color=None):
    return getattr(ConsoleColors, color, "\033[94m")


# def log_failed_img(img_path, img_url):
#     from app.shared.const import IMG_LOG
#     if not os.path.isfile(IMG_LOG):
#         f = open(IMG_LOG, "x")
#         f.close()
#
#     with open(IMG_LOG, "a") as f:
#         f.write(f"{img_path} {img_url}\n")


base_logger = logging.getLogger("DEMO_API")


class TqdmProgress(tqdm):
    # __init__ that simply extracts a vc_callback from the kwargs
    def __init__(self, iterable, *args, **kwargs):
        self._vc_progress_callback = kwargs.pop("vc_progress_callback", None)
        self._vc_close_callback = kwargs.pop("vc_end_callback", None)
        super().__init__(iterable, *args, **kwargs)

    # Overwrite update to call the callback
    def update(self, n=1):
        displayed = super().update(n)
        if displayed and self._vc_progress_callback is not None:
            self._vc_progress_callback(self)

    # Overwrite close to call the callback
    def close(self):
        super().close()
        if self._vc_close_callback is not None:
            self._vc_close_callback(self)


class LoggerHelper:
    def __init__(self):
        raise ValueError("This class is not meant to be instanced")

    @staticmethod
    def info(*s, **kwargs) -> None:
        base_logger.info(" ".join(str(p) for p in s))

    @staticmethod
    def warning(*s, exception: Optional[Exception] = None, **kwargs) -> None:
        text = " ".join(str(p) for p in s)

        if exception:
            text += f"\n {traceback.format_exc(limit=1)}"

        base_logger.warning(text)

    @staticmethod
    def error(*s, exception: Optional[Exception] = None, **kwargs):
        text = " ".join(str(p) for p in s)

        if exception:
            text += f"\n {traceback.format_exc()}"

        base_logger.error(text)

    @classmethod
    def progress(cls, current=0, total=None, title="", **kwargs) -> None:
        base_logger.info(f"Progress {title} {current}/{total}")

    @classmethod
    def iterate(
        cls,
        iterable: Iterable[T],
        title: str = "",
        total: Optional[int] = None,
        rate_limit: float = 1.0,
    ) -> Iterable[T]:
        cls.progress(0, total, title=title)

        def progress_callback(prog: TqdmProgress):
            cls.progress(prog.n, prog.total, title)

        def end_callback(prog: TqdmProgress):
            base_logger.info(f"End {title}")

        return TqdmProgress(
            iterable,
            vc_progress_callback=progress_callback,
            vc_end_callback=end_callback,
            desc=title,
            mininterval=rate_limit,
            total=total,
        )


TLoggerHelper = Type[LoggerHelper]
TJobLogger = TypeVar("TJobLogger", bound="JobLogger")
TLogger = Union[TJobLogger, TLoggerHelper]


class JobLogger:
    LOGGERS: Dict[str, "JobLogger"] = {}

    def __init__(self, message: dramatiq.Message, description=None):
        self._errors = []
        self._warnings = []
        self._grouped_warnings = {}
        self._progress = {}
        self._spawned = []
        self._forwarded = {}
        self._latest_infos = []
        self._message = message
        self._id = message.message_id
        self.description = description
        self._result_backends = []

    def register_backend(self, backend: ResultBackend):
        self._result_backends.append(backend)

    @staticmethod
    def getLoggingRedisKey(message_id: str):
        return f"job:{message_id}"

    @classmethod
    def getLogger(cls: Type[TJobLogger], create: bool = False) -> TLogger:
        """
        Return or create the appropriate logger for the job
        """
        current_task = CurrentMessage.get_current_message()
        if not current_task:
            return LoggerHelper

        job_id = current_task.message_id

        if create and job_id not in cls.LOGGERS:
            cls.LOGGERS[job_id] = cls(current_task)

        return cls.LOGGERS[job_id]

    @classmethod
    def clearLogger(
        cls: Type[TJobLogger], job_id: Optional[str] = None
    ) -> Optional[TLogger]:
        current_task = CurrentMessage.get_current_message()
        if job_id is None:
            if not current_task:
                return None

            job_id = current_task.message_id

        if job_id in cls.LOGGERS:
            return cls.LOGGERS.pop(job_id)

        return None

    def terminate(self) -> Dict[str, str]:
        self.clearLogger(self._id)
        return self.get_state(with_warnings=True)

    def get_state(self, with_warnings: bool = False):
        state = {"id": self._id}

        if self.description:
            state["description"] = self.description

        if self._spawned:
            state["spawned"] = list(self._spawned)

        if self._errors:
            state["errors"] = self._errors

        if self._progress:
            state["progress"] = list(self._progress.values())

        if self._forwarded:
            state["forwarded"] = list(self._forwarded.values())

        if self._latest_infos:
            state["infos"] = self._latest_infos

        if with_warnings:
            warnings = self._warnings
            if self._grouped_warnings:
                for collapse, ws in self._grouped_warnings.items():
                    warnings = [
                        f"{len(ws)} {collapse} warnings. Examples of such warning messages:\n\n{ws[0]}\n{ws[1]}\n{ws[2]}"
                    ] + warnings
            if warnings:
                state["warnings"] = warnings

        return state

    def _send_state(self, state: str = "PROGRESS", with_warnings: bool = False) -> None:
        """
        Update the state of the current task
        Skips warnings unless otherwise specified (they can be many)
        """
        to_send = {"status": state, **self.get_state(with_warnings=with_warnings)}

        for backend in self._result_backends:
            backend.store_result(self._message, to_send, 60 * 60 * 24 * 1000)

    def info(self, *s, **kwargs):
        text = " ".join(str(k) for k in s)
        self._latest_infos.append(text)
        self._latest_infos = self._latest_infos[-10:]
        self._send_state(with_warnings=False)
        LoggerHelper.info(*s, **kwargs)

    def warning(
        self,
        *s,
        collapse: Optional[str] = None,
        exception: Optional[Exception] = None,
        send: bool = False,
    ) -> None:
        text = " ".join(str(k) for k in s)

        if exception:
            text += "\n " + traceback.format_exc(limit=1)

        LoggerHelper.warning(*s, exception=exception)

        if collapse:
            self._grouped_warnings.setdefault(collapse, [])
            self._grouped_warnings[collapse].append(text)
        else:
            self._warnings.append(text)

        if send:
            self._send_state(with_warnings=True)

    def error(self, *s, exception: Optional[Exception] = None) -> None:
        text = " ".join(str(k) for k in s)

        if exception:
            text += "\n " + traceback.format_exc()

        LoggerHelper.error(*s, exception=exception)

        self._errors.append(text)
        self._send_state()

    def progress(
        self,
        current=0,
        total=None,
        title="",
        key: Optional[str] = None,
        end=False,
        print: bool = False,
        send: bool = True,
        **kwargs,
    ) -> None:
        if key is None:
            key = title

        if end:
            try:
                del self._progress[key]
            except KeyError:
                pass
            if send:
                self._send_state(with_warnings=False)
            return

        self._progress[key] = {
            "current": current,
            "total": total,
            "context": title,
        }

        if print:
            LoggerHelper.progress(current, total, title=title, **kwargs)

        if send:
            self._send_state(with_warnings=False)

    def iterate(
        self,
        iterable: Iterable[T],
        title: str = "",
        total: Optional[int] = None,
        rate_limit: float = 1.0,
    ) -> TqdmProgress:
        """
        Monitor the progress of iterating an iterable (through tqdm)
        """

        def progress_callback(prog: TqdmProgress):
            self.progress(prog.n, prog.total, title=title, key=prog.pos)

        def end_callback(prog: TqdmProgress):
            self.progress(end=True, key=prog.pos)

        return TqdmProgress(
            iterable,
            vc_progress_callback=progress_callback,
            vc_end_callback=end_callback,
            desc=title,
            mininterval=rate_limit,
            total=total,
        )


def notifying(func: Optional[Callable[..., Any]] = None) -> Callable[..., Any]:
    """
    A decorator to notify the task of the progress of a function
    """

    def wrapper(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def execute(*args, notify_url: Optional[str] = None, **kwargs):
            logger = JobLogger.getLogger(create=True)
            logger.info(f"Starting task {func.__name__}")
            current_task_id = getattr(logger, "_id", None)

            def notify(event: str, **data):
                if notify_url:
                    requests.post(
                        notify_url,
                        json={"event": event, "tracking_id": current_task_id, **data},
                    )

            try:
                notify("STARTED")
                result = func(*args, **kwargs, logger=logger)
                notify("SUCCESS", success=True, output=result)

                return result
            except Exception as e:
                logger.error(f"Error in task {func.__name__}", exception=e)

                try:
                    notify("ERROR", error=traceback.format_exc())
                except Exception as e:
                    logger.error("Error while notifying", exception=e)

        return execute

    if func is None:
        return wrapper

    return wrapper(func)


class LoggedResults(Results):
    def before_process_message(self, broker, message):
        # store a result saying Progress
        store_results, result_ttl = self._lookup_options(broker, message)
        if store_results:
            logger = JobLogger.getLogger(create=True)
            logger.register_backend(self.backend)


def console(msg, color="bold", e: Optional[Exception] = None):
    print(f"\n\n[{get_time()}]\n{get_color(color)}{pprint(msg)}{ConsoleColors.end}\n")
    if e:
        print(
            f"\nStack Trace:\n{get_color('red')}{traceback.format_exc()}{ConsoleColors.end}\n"
        )


class LoggingTaskMixin:
    def __init__(self, logger: TLogger, *args, **kwargs):
        self.jlogger = logger
        super().__init__(*args, **kwargs)

    def print_and_log(self, s, e: Optional[Exception] = None, **kwargs) -> None:
        console(s, e=e, **kwargs)
        if e:
            self.jlogger.error(s, exception=e)
            return
        self.jlogger.info(s)

    def print_and_log_info(self, s) -> None:
        console(s)
        self.jlogger.info(s)

    def print_and_log_warning(self, s) -> None:
        console(s, color="yellow")
        self.jlogger.warning(s)

    def print_and_log_error(self, s, e: Exception) -> None:
        console(s, color="red", e=e)
        self.jlogger.error(s, exception=e)

    def run_task(self, *args, **kwargs):
        result = super().run_task(*args, **kwargs)
        return result
