import io
import zipfile

import requests
import os
import torch

from pathlib import Path
from typing import Optional

from ..shared.utils.fileutils import send_update, download_file, has_content
from ..shared.utils.img import download_img, get_img_paths
from ..shared.utils.logging import LoggingTaskMixin
from .const import IMG_PATH, MAX_SIZE, MODEL_CONFIG, MODEL_CHECKPOINT, VEC_RESULTS_PATH

from .lib.src import build_model_main
from .lib.src.inference import (
    set_config,
    preprocess_img,
    generate_prediction,
    postprocess_preds,
    save_pred_as_svg,
)


def load_model(model_checkpoint_path=MODEL_CHECKPOINT, model_config_path=MODEL_CONFIG):
    # TODO allow for multiple models
    if not os.path.exists(model_checkpoint_path):
        download_file(
            "https://huggingface.co/seglinglin/Historical-Diagram-Vectorization/resolve/main/checkpoint0045.pth?download=true",
            model_checkpoint_path,
        )
        download_file(
            "https://huggingface.co/seglinglin/Historical-Diagram-Vectorization/resolve/main/config_cfg.py?download=true",
            model_config_path,
        )

    config = set_config(model_config_path)
    model, _, postprocessors = build_model_main(config)
    checkpoint = torch.load(model_checkpoint_path, map_location="cuda")
    model.load_state_dict(checkpoint["model"])
    return model, postprocessors


class ComputeVectorization:
    def __init__(
        self,
        experiment_id: str,
        documents: dict,
        model: Optional[str] = None,
        notify_url: Optional[str] = None,
        tracking_url: Optional[str] = None,
    ):
        self.experiment_id = experiment_id
        self.documents = documents
        self.model = model
        self.notify_url = notify_url
        self.tracking_url = tracking_url
        self.client_id = "default"
        self.imgs = []

    def run_task(self):
        pass

    def check_dataset(self):
        # TODO add more checks
        if len(list(self.documents.keys())) == 0:
            return False
        return True

    def task_update(self, event, message=None):
        if self.tracking_url:
            send_update(self.experiment_id, self.tracking_url, event, message)
            return True
        else:
            return False


class LoggedComputeVectorization(LoggingTaskMixin, ComputeVectorization):
    def run_task(self):
        if not self.check_dataset():
            self.print_and_log_warning(f"[task.vectorization] No documents to download")
            self.task_update(
                "ERROR", f"[API ERROR] No documents to download in dataset"
            )
            return

        error_list = []

        try:
            self.task_update("STARTED")

            model, postprocessors = load_model()
            model.eval()

            for doc_id, document in self.documents.items():
                self.print_and_log(
                    f"[task.vectorization] Vectorization task triggered for {doc_id} !"
                )
                self.download_document(doc_id, document)

                output_dir = VEC_RESULTS_PATH / doc_id
                os.makedirs(output_dir, exist_ok=True)

                for path in get_img_paths(IMG_PATH / doc_id, (".jpg", ".jpeg")):
                    orig_img, tr_img = preprocess_img(path)
                    preds = generate_prediction(orig_img, tr_img, model, postprocessors)
                    preds = postprocess_preds(preds, orig_img.size)
                    save_pred_as_svg(
                        path,
                        img_name=os.path.splitext(os.path.basename(path))[0],
                        img_size=orig_img.size,
                        pred_dict=preds,
                        pred_dir=output_dir,
                    )

                self.send_zip(doc_id)

            self.task_update("SUCCESS", error_list if error_list else None)

        except Exception as e:
            self.print_and_log(f"Error when computing vectorization", e=e)
            self.task_update("ERROR", f"[API ERROR] Vectorization task failed: {e}")

    def download_document(self, doc_id, document):
        self.print_and_log(
            f"[task.vectorization] Downloading {doc_id} images...", color="blue"
        )
        if has_content(f"{IMG_PATH}/{doc_id}/", file_nb=len(document.items())):
            self.print_and_log(
                f"[task.vectorization] {doc_id} already downloaded. Skipping..."
            )
            return

        for img_name, img_url in document.items():
            try:
                download_img(img_url, doc_id, img_name, IMG_PATH, MAX_SIZE)

            except Exception as e:
                self.print_and_log(
                    f"[task.vectorization] Unable to download image {img_name}", e
                )

    def send_zip(self, doc_id):
        """
        Zips the vectorization results and sends the zip to the notify_url
        """
        output_dir = VEC_RESULTS_PATH / doc_id
        try:
            self.print_and_log(
                f"[task.vectorization] Zipping directory {output_dir}", color="blue"
            )

            zip_buffer = io.BytesIO()
            try:
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
                    for root, _, files in os.walk(output_dir):
                        for file in files:
                            file_path = Path(root) / file
                            arcname = file_path.relative_to(output_dir)
                            zipf.write(file_path, arcname)
                zip_buffer.seek(0)  # Rewind buffer to the beginning
            except Exception as e:
                self.print_and_log(
                    f"[task.vectorization] Failed to zip directory {output_dir}", e
                )
                return

            response = requests.post(
                url=self.notify_url,
                files={"file": (f"{doc_id}.zip", zip_buffer, "application/zip")},
                data={"experiment_id": self.experiment_id, "model": self.model},
            )

            if response.status_code == 200:
                self.print_and_log(
                    f"[task.vectorization] Zip successfully sent to {self.notify_url}",
                    color="yellow",
                )
                return

            self.print_and_log(
                f"[task.vectorization] Failed to send zip to {self.notify_url}. Status code: {response.status_code}",
                color="red",
            )

        except Exception as e:
            self.print_and_log(
                f"[task.vectorization] Failed to zip and send directory {output_dir}", e
            )
