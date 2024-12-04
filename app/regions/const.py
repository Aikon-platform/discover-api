"""
Module-specific constants for the regions app
"""
from pathlib import Path
from ..shared.utils.fileutils import create_dirs_if_not
from ..config.base import ENV, BASE_DIR, XACCEL_PREFIX, API_DATA_FOLDER

DEMO_NAME = "regions"

# Path to regions/ folder
DEMO_DIR = BASE_DIR / "app" / DEMO_NAME
LIB_PATH = DEMO_DIR / "lib"

EXT_QUEUE = "queue0"  # see docker-confs/supervisord.conf

EXT_DATA_FOLDER = API_DATA_FOLDER / DEMO_NAME
EXT_XACCEL_PREFIX = Path(ENV("EXT_XACCEL_PREFIX", default="/media/regions-results"))

# IMG_PATH = EXT_DATA_FOLDER / "documents" / "images"
ANNO_PATH = EXT_DATA_FOLDER / "results"
MODEL_PATH = EXT_DATA_FOLDER / "models"

# create_dirs_if_not([IMG_PATH, ANNO_PATH, MODEL_PATH])
create_dirs_if_not([ANNO_PATH, MODEL_PATH])

# TODO retrieve available model instead
DEFAULT_MODEL = "best_eida.pt"
