"""
Many functions to manipulate files and directories
"""

import os
import shutil

import requests
import mimetypes

from datetime import datetime
from os.path import exists
from pathlib import Path
from slugify import slugify
from typing import Union, Optional, Set, List, Tuple, Generator, Iterable
from flask import Response
from stat import S_IFREG
from stream_zip import ZIP_32, stream_zip
import re

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


def is_too_old(filepath: Path, max_days: int = 30) -> bool:
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


def check_dir(path: TPath) -> bool:
    """
    Check if a directory exists, if not create it

    Returns True if the directory existed, False otherwise
    """
    path = Path(path)
    if not path.exists():
        create_dir(path)
        return False
    return True


def create_dirs_if_not(paths: List[TPath]) -> List[TPath]:
    """
    Create directories if they do not exist
    """
    for path in paths:
        check_dir(path)
    return paths


def create_file_if_not(path: TPath) -> Path:
    """
    Create a file if it does not exist, return the path
    """
    path = Path(path)
    if not path.exists():
        path.touch(mode=0o666)
    return path


def create_files_if_not(paths: List[TPath]) -> List[TPath]:
    """
    Create files if they do not exist
    """
    for path in paths:
        create_file_if_not(path)
    return paths


def delete_path(path: TPath) -> bool:
    """
    Delete a file or directory

    Returns True if the path existed and was deleted, False otherwise
    """
    try:
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
    except Exception as e:
        return False
    return True


def clear_dir(parent_dir: TPath, path_to_clear: str = "*", file_to_check: str = None,
              force_deletion: bool = False) -> int:
    """
    Clear a directory of files older than a default number of days
    For folders, the first file (or file_to_check) is checked for age

    Args:
        parent_dir: The parent directory to clear
        path_to_clear: The path to clear (default: "*")
        file_to_check: The file in the directory whose age is checked (default: None)
        force_deletion: If False, only delete files older than a default number of days (default: False)
    """
    cleared = 0
    if not parent_dir:
        return cleared

    for path in parent_dir.glob(path_to_clear):
        file = path
        if path.is_dir():
            file = path / file_to_check if file_to_check else next(path.iterdir())

        if force_deletion or is_too_old(file):
            cleared += 1 if delete_path(path) else 0
    return cleared


def get_file_ext(filepath: TPath) -> str:
    """
    Get the extension of a file without the dot
    """
    path, ext = os.path.splitext(filepath)
    _, filename = os.path.split(path)
    return filename if ext else None, ext[1:] if ext else None


def sanitize_url(string: str) -> str:
    """
    Sanitize a URL to remove spaces
    """
    return string.replace(" ", "+").replace(" ", "+")


def sanitize_str(string: str) -> str:
    """
    Sanitize a URL string to make it a valid filename
    (remove http, https, www, /, ., :, spaces)
    """
    return re.sub(r"^https?\:\/\/|www\.|\.|\:|\s", "", string.strip()).replace("/", "^").replace(" ", "_")


def empty_file(path: TPath) -> None:
    """
    Clear the content of a file if it exists
    """
    if exists(path):
        open(path, "w").close()


def file_age(path: TPath = None) -> int:
    """
    Calculates and returns the age of a file in days based on its last modification time.

    :param path: Path to the file (default: __file__)
    """
    if path is None:
        path = __file__
    dt = datetime.now() - datetime.fromtimestamp(Path(path).stat().st_mtime)  # delta
    return dt.days  # + dt.seconds / 86400  # fractional days


def delete_directory(doc_dir):
    """
    Delete the directory

    Args:
        doc_dir: Directory to delete

    Returns:
        True if the directory existed and was deleted, False otherwise
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


def download_file(url: str, filepath: TPath) -> None:
    """
    Download a file from a URL and save it to disk
    """
    response = requests.get(url)
    if response.status_code == 200:
        with open(filepath, "wb") as file:
            file.write(response.content)
        return
    console(f"Failed to download the file. Status code: {response.status_code}", "red")


def get_all_files(
    directory: str | Path,
    extensions: Optional[Set[str]] = None,
    exclude_dirs: Optional[Set[str]] = None,
    absolute_path: bool = False,
) -> List[Path]:
    """
    Get all files in a directory and its subdirectories.

    Args:
        directory: Base directory path
        extensions: Optional set of extensions to filter files (e.g. {'.txt', '.pdf'})
        exclude_dirs: Optional set of directory names to exclude from search
        absolute_path: Return absolute path

    Returns:
        List of Path objects for all matching files
    """
    if isinstance(directory, str):
        directory = Path(directory)

    if absolute_path:
        directory = directory.resolve()

    if not directory.exists():
        return []

    files = []
    try:
        for item in directory.rglob("*"):
            if exclude_dirs and any(p.name in exclude_dirs for p in item.parents):
                continue

            if item.is_file():
                if extensions is None or item.suffix.lower() in extensions:
                    files.append(item)
    except PermissionError:
        # no permission to directories
        pass

    return sorted(files)


def zip_on_the_fly(files: List[Tuple[str, TPath]]) -> Iterable[bytes]:
    """
    Zip files on the fly

    Args:
        files: List of tuples (filename, path)
    """

    def contents(path: TPath) -> Generator[bytes, None, None]:
        with open(path, 'rb') as f:
            while chunk := f.read(65536):
                yield chunk

    def iter_files() -> Generator[Tuple[str, int, int, int, Generator[bytes, None, None]], None, None]:
        for name, path in files:
            if not os.path.exists(path):
                continue
            dt = datetime.fromtimestamp(os.path.getmtime(path))
            yield (name, dt, S_IFREG | 0o600, ZIP_32, contents(path))

    return stream_zip(iter_files())


def serializer(obj):
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Type {type(obj)} is not JSON serializable")
