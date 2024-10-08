from pathlib import Path
from ..shared.utils.fileutils import create_file_if_not
from ..config.base import BASE_DIR

DIR_NAME = "shared"

# Path to shared/ folder
SHARED_DIR = BASE_DIR / "app" / DIR_NAME
UTILS_DIR = SHARED_DIR / "utils"

IMG_LOG = Path(f"{SHARED_DIR}/img.log")

create_file_if_not(IMG_LOG)
