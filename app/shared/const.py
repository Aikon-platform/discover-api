"""
Various constants and paths, derived from the config
"""

from pathlib import Path
from .utils.fileutils import create_file_if_not, create_dirs_if_not
from ..config import BASE_DIR, API_DATA_FOLDER

DIR_NAME = "shared"
DEMO_NAME = DIR_NAME

# Path to shared/ folder
SHARED_DIR = BASE_DIR / "app" / DIR_NAME
EXT_DATA_FOLDER = API_DATA_FOLDER / DIR_NAME

UTILS_DIR = SHARED_DIR / "utils"

IMG_PATH = EXT_DATA_FOLDER / "images"
DATASETS_PATH = EXT_DATA_FOLDER / "datasets"
DOCUMENTS_PATH = EXT_DATA_FOLDER / "documents"

create_dirs_if_not([IMG_PATH])
