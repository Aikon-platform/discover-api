from ..shared.utils.fileutils import create_dirs_if_not, download_model_if_not
from ..config.base import BASE_DIR, XACCEL_PREFIX, API_DATA_FOLDER
import torch

DEMO_NAME = "vectorization"

# Path to extraction/ folder
DEMO_DIR = BASE_DIR / "app" / DEMO_NAME
LIB_PATH = DEMO_DIR / "lib"

VEC_QUEUE = "queue4"  # see docker-confs/supervisord.conf

VEC_DATA_FOLDER = API_DATA_FOLDER / DEMO_NAME
VEC_XACCEL_PREFIX = f"{XACCEL_PREFIX}/{DEMO_NAME}"

VEC_RESULTS_PATH = VEC_DATA_FOLDER / "results"
MODEL_PATH = VEC_DATA_FOLDER / "models"

create_dirs_if_not([VEC_RESULTS_PATH, MODEL_PATH])
torch.hub.set_dir(f"{BASE_DIR}/.cache")

MODEL_CHECKPOINT = MODEL_PATH / "checkpoint0045.pth"
MODEL_CONFIG = MODEL_PATH / "config_cfg.py"

download_model_if_not(
    {
        "repo_id": "seglinglin/Historical-Diagram-Vectorization",
        "filename": "checkpoint0045.pth",
    },
    MODEL_PATH / "checkpoint0045.pth",
)

download_model_if_not(
    {
        "repo_id": "seglinglin/Historical-Diagram-Vectorization",
        "filename": "config_cfg.py",
    },
    MODEL_PATH / "config_cfg.py",
)

DEFAULT_MODEL_INFOS = {
    "checkpoint0045": {
        "name": "Historical Diagram vectorization model",
        "model": "checkpoint0045",
        "desc": "DINO-DETR model finetuned on historical diagrams.",
    },
}

MAX_SIZE = 244
