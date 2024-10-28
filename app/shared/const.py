from pathlib import Path
from ..shared.utils.fileutils import create_file_if_not, create_dirs_if_not
from ..config.base import BASE_DIR, API_DATA_FOLDER

DIR_NAME = "shared"

# Path to shared/ folder
SHARED_DIR = BASE_DIR / "app" / DIR_NAME
EXT_DATA_FOLDER = API_DATA_FOLDER / DIR_NAME
UTILS_DIR = SHARED_DIR / "utils"

IMG_LOG = Path(f"{SHARED_DIR}/img.log")
IMG_PATH = EXT_DATA_FOLDER / "images"

create_file_if_not(IMG_LOG)
create_dirs_if_not([IMG_PATH])
