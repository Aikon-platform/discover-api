"""
Module-specific constants for the regions app
"""
from pathlib import Path

from ..shared.utils.fileutils import create_dirs_if_not, download_model_if_not
from ..config.base import ENV, BASE_DIR, API_DATA_FOLDER

DEMO_NAME = "regions"

# Path to regions/ folder
DEMO_DIR = BASE_DIR / "app" / DEMO_NAME
LIB_PATH = DEMO_DIR / "lib"

EXT_QUEUE = "queue0"  # see docker-confs/supervisord.conf

EXT_DATA_FOLDER = API_DATA_FOLDER / DEMO_NAME
EXT_XACCEL_PREFIX = Path(ENV("EXT_XACCEL_PREFIX", default="/media/regions-results"))

MODEL_PATH = EXT_DATA_FOLDER / "models"

create_dirs_if_not([MODEL_PATH])

# TODO retrieve available model instead
download_model_if_not(
    "https://huggingface.co/seglinglin/Historical-Illustration-Extraction/resolve/main"
    "/illustration_extraction.pt?download=true",
    MODEL_PATH / "illustration_extraction.pt",
)
download_model_if_not(
    "https://huggingface.co/seglinglin/Historical-Illustration-Extraction/resolve/main"
    "/diagram_extraction.pt?download=true",
    MODEL_PATH / "diagram_extraction.pt",
)
DEFAULT_MODEL = "illustration_extraction.pt"

DEFAULT_MODEL_INFOS = {
    "illustration_extraction": {
        "name": "Illustration extraction",
        "model": "illustration_extraction",
        "desc": "YOLO model fine-tuned on historical illustrations.",
    },
    "diagram_extraction": {
        "name": "Diagram extraction",
        "model": "diagram_extraction",
        "desc": "YOLO model fine-tuned on historical diagrams.",
    },
}
