import json
import os
from pathlib import Path
from typing import Optional
import requests
import torch
from torchvision import transforms
from PIL import Image

from .const import DEFAULT_MODEL, ANNO_PATH, MODEL_PATH, IMG_PATH
from .lib.extract import YOLOExtractor, FasterRCNNExtractor
from ..shared.tasks import LoggedTask
from ..shared.dataset import Document, Dataset


class ExtractRegions(LoggedTask):
    def __init__(
        self,
        dataset: Dataset,
        model: Optional[str] = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.dataset = dataset
        self._model = model
        self._extraction_model: Optional[str] = None
        self.result_dir = Path()
        self.annotations = {}

    def initialize(self):
        if self.model.startswith(("rcnn", "fasterrcnn")):
            self.extractor = FasterRCNNExtractor(self.weights)
        else:
            self.extractor = YOLOExtractor(self.weights)

    def terminate(self):
        del self.extractor

    @property
    def model(self) -> str:
        return DEFAULT_MODEL if self._model is None else self._model

    @property
    def weights(self) -> Path:
        return MODEL_PATH / self.model

    @property
    def extraction_model(self) -> str:
        if self._extraction_model is None:
            self._extraction_model = self.model.split(".")[0]
        return self._extraction_model

    def check_doc(self) -> bool:
        # TODO improve check regarding dataset content
        if not self.dataset.documents:
            return False
        return True

    def send_annotations(
        self,
        experiment_id: str,
        annotation_file: Path,
    ) -> bool:
        if not self.notify_url:
            self.error_list.append("Notify URL not provided")
            return True

        with open(annotation_file, "r") as f:
            annotation_file = f.read()

        response = requests.post(
            url=f"{self.notify_url}/{self.dataset.uid}",
            files={"annotation_file": annotation_file},
            data={
                "model": self.extraction_model,
                "experiment_id": experiment_id,
            },
        )
        response.raise_for_status()
        return True

    def process_img(
        self,
        img_path: Path,
        extraction_ref: str,
        img_number: int
    ) -> bool:
        filename = img_path.name
        try:
            self.print_and_log(f"====> Processing {filename} 🔍")
            anno = self.extractor.extract_one(img_path)
            self.annotations[extraction_ref].append(anno)
            return True
        except Exception as e:
            self.handle_error(
                f"Error processing image {filename}",
                exception=e
            )
            return False

    def process_doc_imgs(
        self,
        doc: Document,
        extraction_ref: str
    ) -> bool:
        images = doc.list_images()
        self.annotations[extraction_ref] = []
        try:
            for i, image in enumerate(images, 1):
                success = self.process_img(image, extraction_ref, i)
                if not success:
                    self.handle_error(f"Failed to process {image}")
        except Exception as e:
            self.handle_error(
                f"Error processing images for {doc.uid}",
                exception=e
            )
            return False
        return True

    def process_doc(
        self,
        doc: Document
    ) -> bool:
        try:
            self.print_and_log(f"[task.extract_regions] Downloading {doc.uid}...")

            # image_dir, dataset_ref = download_dataset(
            #     doc_url,
            #     datasets_dir_path=IMG_PATH,
            #     dataset_dir_name=doc_id,
            # )

            doc.download()
            self.result_dir = doc.annotations_path
            os.makedirs(self.result_dir, exist_ok=True)

            extraction_ref = f"{self.extraction_model}_{self.experiment_id}"
            annotation_file = self.result_dir / f"{extraction_ref}.txt"
            with open(annotation_file, 'w'):
                pass

            self.print_and_log(f"DETECTING VISUAL ELEMENTS FOR {doc.uid} 🕵️")
            success = self.process_doc_imgs(doc, extraction_ref)
            if success:
                with open(self.result_dir / f"{extraction_ref}.json", 'w') as f:
                    json.dump(self.annotations[extraction_ref], f, indent=2)

                success = self.send_annotations(
                    self.experiment_id,
                    annotation_file,
                )

            return success
        except Exception as e:
            self.handle_error(
                f"Error processing document {doc.uid}",
                exception=e
            )
            return False

    def run_task(self) -> bool:
        if not self.check_doc():
            self.print_and_log_warning(
                "[task.extract_regions] No dataset to annotate"
            )
            self.task_update(
                "ERROR",
                f"[API ERROR] Failed to download dataset for {self.dataset}",
            )
            return False

        self.task_update("STARTED")
        self.print_and_log(
            f"[task.extract_regions] Extraction task triggered with {self.model}!"
        )

        try:
            self.initialize()
            all_successful = True
            for doc in self.dataset.documents:
                success = self.process_doc(doc)
                all_successful = all_successful and success

            status = "SUCCESS" if all_successful else "ERROR"
            self.print_and_log(f"[task.extract_regions] Task completed with status: {status}")
            self.task_update(status, self.error_list if self.error_list else None)
            return all_successful
        except Exception as e:
            self.handle_error(str(e))
            self.task_update("ERROR", self.error_list)
            return False
        finally:
            self.terminate()
