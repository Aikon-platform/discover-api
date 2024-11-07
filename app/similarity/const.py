from ..shared.utils.fileutils import create_dirs_if_not
from ..config.base import BASE_DIR, XACCEL_PREFIX, API_DATA_FOLDER

DEMO_NAME = "similarity"

# Path to similarity/ folder
DEMO_DIR = BASE_DIR / "app" / DEMO_NAME
LIB_PATH = DEMO_DIR / "lib"

SIM_QUEUE = "queue1"  # see docker-confs/supervisord.conf

SIM_DATA_FOLDER = API_DATA_FOLDER / DEMO_NAME
SIM_XACCEL_PREFIX = f"{XACCEL_PREFIX}/{DEMO_NAME}"
SIM_RESULTS_PATH = SIM_DATA_FOLDER / "results"

IMG_PATH = SIM_DATA_FOLDER / "documents"
MODEL_PATH = SIM_DATA_FOLDER / "models"
SCORES_PATH = SIM_RESULTS_PATH
FEATS_PATH = SIM_DATA_FOLDER / "feats"

create_dirs_if_not([IMG_PATH, MODEL_PATH, SCORES_PATH, FEATS_PATH])

