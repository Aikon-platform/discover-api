"""
The main routes that handle the API requests regarding starting and monitoring tasks.
"""

import functools

from flask import request, send_from_directory, jsonify, Request
from slugify import slugify
from dramatiq import Actor, Broker
from dramatiq_abort import abort
from dramatiq.results import ResultMissing, ResultFailure
import traceback
from typing import Tuple

from .utils import hash_str
from .utils.logging import console
from .. import config

from .utils.fileutils import xaccel_send_from_directory


def error_wrapper(func):
    """
    A decorator that catches exceptions and returns them as JSON Flask Response
    """
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


def receive_task(req:Request) -> Tuple[str, str, str, dict]:
    """
    Extracts the parameters from the request and returns them

    :param req: The Flask request object

    :return: The experiment_id, notify_url, tracking_url, and the parameters
    """
    param = req.get_json() if req.is_json else req.form.to_dict()
    if not param:
        raise ValueError("No data in request: Task aborted!")

    console(param, color="cyan")

    experiment_id = param.get('experiment_id')
    tracking_url = param.get("tracking_url")
    # AIKON => "callback" / DISCOVER-DEMO => "notify_url" (TODO unify)
    notify_url = param.get('notify_url') or param.get('callback')

    # task_kwargs = {}
    # for param_name in additional_params:
    #     task_kwargs[param_name] = param.get(param_name, None)

    return experiment_id, notify_url, tracking_url, param


def start_task(task_fct:Actor, experiment_id:str, task_kwargs: dict) -> dict:
    """
    Start a new task

    :param task_fct: The task function to run
    :param experiment_id: The ID of the experiment
    :param task_kwargs: The parameters of the task

    :return: A dictionary containing the tracking_id and the experiment_id
    """
    task = task_fct.send(experiment_id=experiment_id, **task_kwargs)

    return {
        "tracking_id": task.message_id,
        "experiment_id": experiment_id,
    }


def cancel_task(tracking_id: str) -> dict:
    """
    Cancel a task

    :param tracking_id: The ID of the task to cancel

    :return: A dictionary containing the tracking_id
    """
    abort(tracking_id)

    return {"tracking_id": tracking_id}


def status(tracking_id: str, task_fct: Actor) -> dict:
    """
    Get the status of a task

    :param tracking_id: The ID of the task
    :param task_fct: The task function

    :return: A dictionary containing the tracking_id and the log of the task
    """
    try:
        log = task_fct.message().copy(message_id=tracking_id).get_result()
    except ResultMissing:
        log = {"status": "PENDING", "infos": ["Task still running"]}
    except ResultFailure as e:
        log = {
            "status": "ERROR",
            "infos": [f"Error: Actor raised {e.orig_exc_type} ({e.orig_exc_msg})"],
        }
    return {
        "tracking_id": tracking_id,
        "log": log,
    }


def result(tracking_id: str, results_dir: str, xaccel_prefix: str, extension: str="zip"):
    """
    Get the result of a task

    :param tracking_id: The ID of the task
    :param results_dir: The directory where the results are stored
    :param xaccel_prefix: The prefix for the X-Accel-Redirect header
    :param extension: The extension of the result file (without the dot)

    :return: The result file as a Flask response
    """
    if not config.USE_NGINX_XACCEL:
        return send_from_directory(results_dir, f"{slugify(tracking_id)}.{extension}")

    return xaccel_send_from_directory(
        results_dir, xaccel_prefix, f"{slugify(tracking_id)}.{extension}"
    )


def qsizes(broker: Broker) -> dict:
    """
    List the queues of the broker and the number of tasks in each queue

    :param broker: The broker to get the queues from

    :return: A dictionary containing the queues and their sizes, or an error message
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


def monitor(results_dir: str, broker: Broker) -> dict:
    """
    Monitor the app service

    :param results_dir: The directory where the results are stored
    :param broker: The broker to get the queues from

    :return: A dictionary containing the total size of the results directory and the queue sizes
    """
    total_size = 0
    for path in results_dir.glob("**/*"):
        total_size += path.stat().st_size

    return {"total_size": total_size, **qsizes(broker)}
