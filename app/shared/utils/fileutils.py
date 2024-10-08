import os
import shutil

import requests
import mimetypes

from datetime import datetime
from os.path import exists
from pathlib import Path
from slugify import slugify
from typing import Union
from flask import Response

from .logging import console

TPath = Union[str, os.PathLike]


def xaccel_send_from_directory(directory: TPath, redirect: TPath, path: TPath):
    """
    Send a file from a given directory using X-Accel-Redirect
    """
    try:
        path = Path(path)
        directory = Path(directory)
        file_path = directory / path

        assert file_path.is_relative_to(directory)
        redirect_path = Path(redirect) / file_path.relative_to(directory)

        content_length = file_path.stat().st_size
        content_type = mimetypes.guess_type(str(file_path))[0]
        filename = file_path.name

        if not content_length or not content_type or not filename:
            return None

        response = Response()
        response.headers["Content-Length"] = content_length
        response.headers["Content-Type"] = content_type
        response.headers[
            "Content-Disposition"
        ] = f"attachment; filename={slugify(str(filename))}"
        response.headers["X-Accel-Redirect"] = str(redirect_path)
        response.headers["X-Accel-Buffering"] = "yes"
        response.headers["X-Accel-Charset"] = "utf-8"
        return response

    except Exception:
        return None


def is_too_old(filepath: Path, max_days: int = 30):
    """
    Check if a file is older than a given number of days
    """
    try:
        return (
            not filepath.exists()
            or (datetime.now() - datetime.fromtimestamp(filepath.stat().st_mtime)).days
            > max_days
        )
    except Exception:
        return False


def has_content(path, file_nb=None):
    path = Path(path)
    if not os.path.exists(path):
        create_dir(path)
        return False

    nb_of_files = len(os.listdir(path))
    if file_nb:
        return nb_of_files == file_nb
    return nb_of_files != 0


def create_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def check_dir(path):
    path = Path(path)
    if not path.exists():
        create_dir(path)
        return False
    return True


def create_dirs_if_not(paths):
    for path in paths:
        check_dir(path)
    return paths


def create_file_if_not(path):
    path = Path(path)
    if not path.exists():
        path.touch(mode=0o666)
    return path


def create_files_if_not(paths):
    for path in paths:
        create_file_if_not(path)
    return paths


def delete_path(path):
    try:
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
    except Exception as e:
        return False
    return True


def clear_dir(parent_dir, path_to_clear="*", file_to_check=None, condition=None):
    """
    Clear a directory of files older than a given number of days

    """
    cleared = 0
    if not parent_dir:
        return cleared

    for path in parent_dir.glob(path_to_clear):
        file = path
        if path.is_dir():
            file = path / file_to_check if file_to_check else next(path.iterdir())

        condition = condition if condition is not None else is_too_old(file)
        if condition:
            cleared += 1 if delete_path(path) else 0
    return cleared


def get_file_ext(filepath):
    path, ext = os.path.splitext(filepath)
    _, filename = os.path.split(path)
    return filename if ext else None, ext[1:] if ext else None


def sanitize_url(string):
    return string.replace(" ", "+").replace(" ", "+")


def sanitize_str(string):
    return (
        string.replace("/", "")
        .replace(".", "")
        .replace("https", "")
        .replace("http", "")
        .replace("www", "")
        .replace(" ", "_")
        .replace(":", "")
    )


def empty_file(string):
    if exists(string):
        open(string, "w").close()


def send_update(experiment_id, tracking_url, event, message):
    response = requests.post(
        url=tracking_url,
        data={
            "experiment_id": experiment_id,
            "event": event,
            "message": message if message else "",
        },
    )
    response.raise_for_status()
    return True


def file_age(path=__file__):
    """
    Calculates and returns the age of a file in days based on its last modification time.
    """
    dt = datetime.now() - datetime.fromtimestamp(Path(path).stat().st_mtime)  # delta
    return dt.days  # + dt.seconds / 86400  # fractional days


def delete_directory(doc_dir):
    """
    Delete the directory corresponding to the document ID to relaunch vectorization
    """
    path = Path(doc_dir)
    try:
        if os.path.isdir(path):
            shutil.rmtree(path)
            return True
        else:
            return False
    except Exception as e:
        console(f"An error has occurred when deleting {doc_dir} directory", e=e)
        return False


def download_file(url, filepath):
    response = requests.get(url)
    if response.status_code == 200:
        with open(filepath, "wb") as file:
            file.write(response.content)
        return
    console(f"Failed to download the file. Status code: {response.status_code}", "red")
