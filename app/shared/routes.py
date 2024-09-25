import functools

from flask import request, send_from_directory, jsonify
from slugify import slugify
from dramatiq_abort import abort
from dramatiq.results import ResultMissing, ResultFailure
import traceback

from .utils import hash_str
from .. import config

from .utils.fileutils import xaccel_send_from_directory


def error_wrapper(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            return jsonify({"error": str(e), "traceback": traceback.format_exc()})

    return wrapped


def get_client_id(func):
    @functools.wraps(func)
    def decorator(*args, **kwargs):
        headers = request.headers
        client_addr = f"{headers.get('User-Agent', '')}_{request.remote_addr}_{headers.get('X-Forwarded-For', '')}"
        if request.method == "POST" and client_addr:
            client_id = str(hash_str(client_addr))[:8]
            return func(client_id, *args, **kwargs)
        else:
            return {"message": "Error when generating the client_id"}, 403

    return decorator


def start_task(task_fct, experiment_id, task_kwargs):
    """
    Start a new task
    """
    task = task_fct.send(experiment_id=experiment_id, **task_kwargs)

    return {
        "tracking_id": task.message_id,
        "experiment_id": experiment_id,
    }


def cancel_task(tracking_id: str):
    """
    Cancel a task
    """
    abort(tracking_id)

    return {"tracking_id": tracking_id}


def status(tracking_id: str, task_fct):
    """
    Get the status of a task
    """
    try:
        log = task_fct.message().copy(message_id=tracking_id).get_result()
    except ResultMissing:
        # TODO check in what cases this happen (only after task execution?)
        log = None
    except ResultFailure as e:
        log = {
            "status": "ERROR",
            "infos": [f"Error: Actor raised {e.orig_exc_type} ({e.orig_exc_msg})"],
        }

    return {
        "tracking_id": tracking_id,
        "log": log,
    }


def result(tracking_id: str, results_dir, xaccel_prefix):
    """
    Get the result of a task
    """
    if not config.USE_NGINX_XACCEL:
        return send_from_directory(results_dir, f"{slugify(tracking_id)}.zip")

    return xaccel_send_from_directory(
        results_dir, xaccel_prefix, f"{slugify(tracking_id)}.zip"
    )


def qsizes(broker):
    """
    List the queues of the broker and the number of tasks in each queue
    """
    try:
        return {
            "queues": {
                q: {"name": q, "size": broker.do_qsize(q)}
                for q in broker.get_declared_queues()
            }
        }
    except AttributeError:
        return {"error": "Cannot get queue sizes from broker"}


def monitor(results_dir, broker):
    # Get the status of the app service
    total_size = 0
    for path in results_dir.glob("**/*"):
        total_size += path.stat().st_size

    return {"total_size": total_size, **qsizes(broker)}
