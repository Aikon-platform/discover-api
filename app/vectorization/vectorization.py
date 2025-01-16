import traceback
import zipfile

import requests
import os
import torch

from pathlib import Path
from typing import Optional

from ..config import BASE_URL
from ..shared.tasks import LoggedTask
from ..shared.utils.fileutils import download_file
from ..shared.utils.logging import TLogger
from .const import (
    MODEL_CONFIG,
    MODEL_CHECKPOINT,
    VEC_RESULTS_PATH,
)  # , IMG_PATH

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


class ComputeVectorization(LoggedTask):
    def __init__(self, documents: dict, model: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.documents = documents
        self.model = model
        self.imgs = []

    def check_dataset(self):
        # TODO add more checks
        if len(list(self.documents.keys())) == 0:
            return False
        return True

    def run_task(self):
        if not self.check_dataset():
            self.print_and_log_warning(f"[task.vectorization] No documents to download")
            raise ValueError(f"[task.vectorization] No documents to download")

        results = {}

        try:
            model, postprocessors = load_model()
            model.eval()

            for doc_id, document in self.documents.items():
                self.print_and_log(
                    f"[task.vectorization] Vectorization task triggered for {doc_id}!"
                )
                self.download_document(doc_id, document)

                output_dir = VEC_RESULTS_PATH / doc_id
                os.makedirs(output_dir, exist_ok=True)

                try:
                    # TODO fix to use dataset path ⚠️⚠️⚠️⚠️
                    # for path in get_img_paths(IMG_PATH / doc_id, (".jpg", ".jpeg")):
                    #     orig_img, tr_img = preprocess_img(path)
                    #     preds = generate_prediction(orig_img, tr_img, model, postprocessors)
                    #     preds = postprocess_preds(preds, orig_img.size)
                    #     save_pred_as_svg(
                    #         path,
                    #         img_name=os.path.splitext(os.path.basename(path))[0],
                    #         img_size=orig_img.size,
                    #         pred_dict=preds,
                    #         pred_dir=output_dir,
                    #     )
                    doc_result = {doc_id: self.create_zip(doc_id)}
                    self.notifier("PROGRESS", doc_result)
                    results.update(doc_result)
                except Exception as e:
                    self.notifier(
                        "ERROR", error=traceback.format_exc(), completed=False
                    )
                    self.error_list.append(f"{e}")

            results.update({"error": self.error_list})
            return results

        except Exception as e:
            self.print_and_log(f"Error when computing vectorization", e=e)
            raise e

    def download_document(self, doc_id, document):
        self.print_and_log(
            f"[task.vectorization] Downloading {doc_id} images...", color="blue"
        )
        # ⚠️⚠️⚠️⚠️ TODO use new dataset way of doing thing
        # if has_content(f"{IMG_PATH}/{doc_id}/", file_nb=len(document.items())):
        #     self.print_and_log(
        #         f"[task.vectorization] {doc_id} already downloaded. Skipping..."
        #     )
        #     return

        # for img_name, img_url in document.items():
        #     # ⚠️⚠️⚠️⚠️ TODO use dataset download
        #     # try:
        #     #     download_img(img_url, doc_id, img_name, IMG_PATH, MAX_SIZE)
        #     #
        #     # except Exception as e:
        #     #     self.print_and_log(
        #     #         f"[task.vectorization] Unable to download image {img_name}", e
        #     #     )

    def create_zip(self, doc_id):
        """
        Creates a zip file containing the vectorization results and saves it to disk
        Returns the path to the created zip file
        """
        output_dir = VEC_RESULTS_PATH / doc_id
        zip_path = output_dir / f"{doc_id}.zip"

        try:
            self.print_and_log(
                f"[task.vectorization] Zipping directory {output_dir}", color="blue"
            )

            try:
                with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                    for root, _, files in os.walk(output_dir):
                        for file in files:
                            # Skip the zip file itself if it exists
                            if file == f"{doc_id}.zip":
                                continue
                            file_path = Path(root) / file
                            arcname = file_path.relative_to(output_dir)
                            zipf.write(file_path, arcname)

                # TODO check if XACCEL_PREFIX work and returns really
                return f"{BASE_URL}/vectorization/{doc_id}/result"

            except Exception as e:
                self.print_and_log(
                    f"[task.vectorization] Failed to create zip file for directory {output_dir}",
                    e,
                )
                raise e

        except Exception as e:
            self.print_and_log(
                f"[task.vectorization] Failed to zip directory {output_dir}", e
            )
            raise e
