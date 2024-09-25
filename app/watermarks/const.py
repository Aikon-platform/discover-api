from pathlib import Path
from ..config.base import ENV, BASE_DIR, API_DATA_FOLDER, XACCEL_PREFIX

DEMO_NAME = "watermarks"

# Path to DEMO_NAME/ folder
DEMO_DIR = BASE_DIR / "app" / DEMO_NAME

WATERMARKS_DATA_FOLDER = API_DATA_FOLDER / DEMO_NAME
WATERMARKS_SOURCES_FOLDER = WATERMARKS_DATA_FOLDER / "sources"
WATERMARKS_RESULTS_FOLDER = WATERMARKS_DATA_FOLDER / "results"
WATERMARKS_TMP_FOLDER = WATERMARKS_DATA_FOLDER / "tmp_queries"
WATERMARKS_XACCEL_PREFIX = f"{XACCEL_PREFIX}/{DEMO_NAME}"

MODEL_PATHS = {
    "detection": WATERMARKS_DATA_FOLDER / "models" / "detection.pth",
    "features": WATERMARKS_DATA_FOLDER / "models" / "features.pth",
}

DEVICE = "cpu"
WATERMARKS_QUEUE = "queue3"  # see docker-confs/supervisord.conf
