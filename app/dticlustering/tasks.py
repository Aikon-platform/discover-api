"""
Tasks for DTI clustering

**train_dti**
This function is a Dramatiq actor that handles the training of a DTI model. 
It downloads the dataset, prepares it, and runs the training process based on the provided parameters.

- experiment_id (str): The ID of the clustering task.
- dataset_id (str): The ID of the dataset.
- dataset_url (str): The URL of the zipped dataset to be downloaded.
- parameters (Optional[dict]): A JSON object containing the training parameters.
- logger (TLogger): A logger object.
- notifier: An optional notifier object.

Returns:

- dict: A dictionary containing the result URL of the trained model.
"""

import dramatiq
from dramatiq.middleware import CurrentMessage
from typing import Optional
from zipfile import ZipFile
from PIL import Image

from .. import config
from .const import DATASETS_PATH, DTI_RESULTS_PATH, DTI_QUEUE
from .training import (
    run_kmeans_training,
    run_sprites_training,
)
from ..shared.utils.download import download_dataset
from ..shared.utils.logging import notifying, TLogger, LoggerHelper


@dramatiq.actor(
    time_limit=1000 * 60 * 60, max_retries=0, store_results=True, queue_name=DTI_QUEUE
)
@notifying
def train_dti(
    experiment_id: str,
    dataset_id: str,
    dataset_url: str,
    parameters: Optional[dict] = None,
    logger: TLogger = LoggerHelper,
    notifier=None,
    **kwargs
):
    """
    Train a DTI model

    Parameters:
    - experiment_id: the ID of the clustering task
    - dataset_id: the ID of the dataset
    - dataset_url: the URL of the zipped dataset to be downloaded
    - parameters: a JSON object containing the training parameters
    - logger: a logger object
    """

    current_task = CurrentMessage.get_current_message()
    current_task_id = current_task.message_id

    result_file = DTI_RESULTS_PATH / f"{current_task_id}.zip"
    result_file.parent.mkdir(parents=True, exist_ok=True)

    # Download and extract dataset to local storage
    dataset_path = DATASETS_PATH / "generic" / dataset_id
    dataset_ready_file = dataset_path / "ready.meta"

    if not dataset_ready_file.exists():
        # dataset_path.mkdir(parents=True, exist_ok=True)
        # dataset_zip_path = dataset_path / "dataset.zip"
        #
        # with requests.get(dataset_url, stream=True) as r:
        #     r.raise_for_status()
        #     with open(dataset_zip_path, "wb") as f:
        #         for chunk in r.iter_content(chunk_size=8192):
        #             f.write(chunk)
        #
        # # Unzip dataset
        # with ZipFile(dataset_zip_path, "r") as zipObj:
        #     zipObj.extractall(dataset_path / "train")
        #
        # dataset_zip_path.unlink()
        download_dataset(dataset_url, datasets_dir_path=DATASETS_PATH, dataset_dir_name=f"generic/{dataset_id}", sub_dir="train")

        # Create ready file
        dataset_ready_file.touch()
    else:
        print("Dataset already ready")
        dataset_ready_file.touch()

    # Start training
    if parameters.get("background_option", "0_dti") == "0_dti":
        # Use DTI clustering
        output_path = run_kmeans_training(experiment_id, dataset_id, parameters, logger)
    else:
        # Use DTI sprites (1_learn_bg / 2_const_bg / 3_learn_fg)
        output_path = run_sprites_training(
            experiment_id, dataset_id, parameters, logger
        )

    # zip results to DTI_RESULTS_PATH
    with ZipFile(result_file, "w") as zipObj:
        for file in output_path.glob("**/*"):
            # if file.suffix == ".pkl": # Don't include the model
            #    continue

            if file.suffix == ".png":  # Convert to jpg if not transparent
                img = Image.open(file)
                if img.mode != "RGBA" and img.format != "JPEG":
                    img.save(file, "JPEG", quality=85)

            zipObj.write(file, file.relative_to(output_path))

    return {"result_url": f"{config.BASE_URL}/clustering/{current_task_id}/result"}
