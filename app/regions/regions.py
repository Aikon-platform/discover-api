import os
from pathlib import Path
from typing import Optional
import requests

from .const import DEFAULT_MODEL, ANNO_PATH, MODEL_PATH, IMG_PATH
from .lib.extract import extract
from ..shared.tasks import LoggedTask
from ..shared.utils.fileutils import empty_file
from ..shared.utils.download import download_dataset
from ..shared.utils.img import get_img_paths


class ExtractRegions(LoggedTask):
    def __init__(
        self,
        documents: dict,
        model: Optional[str] = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.documents = documents
        self._model = model
        self._extraction_model: Optional[str] = None

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
        # TODO improve check regarding documents content
        if not self.documents:
            return False
        return True

    def send_annotations(
        self,
        experiment_id: str,
        annotation_file: Path,
        dataset_ref: str
    ) -> bool:
        if not self.notify_url:
            self.error_list.append("Notify URL not provided")
            return False

        with open(annotation_file, "r") as f:
            annotation_file = f.read()

        response = requests.post(
            url=f"{self.notify_url}/{dataset_ref}",
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
        annotation_file: Path,
        img_number: int
    ) -> bool:
        filename = img_path.name
        try:
            self.print_and_log(f"====> Processing {filename} 🔍")
            extract(
                weights=self.weights,
                source=img_path,
                anno_file=str(annotation_file),
                img_nb=img_number,
            )
            return True
        except Exception as e:
            self.handle_error(
                f"Error processing image {filename}",
                exception=e
            )
            return False

    def process_doc_imgs(
        self,
        dataset_path: Path,
        annotation_file: Path
    ) -> bool:
        images = get_img_paths(dataset_path)
        try:
            for i, image in enumerate(images, 1):
                success = self.process_img(image, annotation_file, i)
                if not success:
                    self.handle_error(f"Failed to process {image}")
        except Exception as e:
            self.handle_error(
                f"Error processing images for {dataset_path}",
                exception=e
            )
            return False
        return True

    def process_doc(
        self,
        doc_id: str,
        doc_url: str,
    ) -> bool:
        try:
            self.print_and_log(f"[task.extract_regions] Downloading {doc_id}...")

            image_dir, dataset_ref = download_dataset(
                doc_url,
                datasets_dir_path=IMG_PATH,
                dataset_dir_name=doc_id,
            )

            annotation_dir = ANNO_PATH / dataset_ref
            os.makedirs(annotation_dir, exist_ok=True)
            # TODO check which name to use / which directory structure is better
            annotation_file = annotation_dir / f"{self.extraction_model}_{self.experiment_id}.txt"
            empty_file(annotation_file)

            self.print_and_log(f"DETECTING VISUAL ELEMENTS FOR {doc_url} 🕵️")
            success = self.process_doc_imgs(IMG_PATH / image_dir, annotation_file)
            if success:
                success = self.send_annotations(
                    self.experiment_id,
                    annotation_file,
                    dataset_ref,
                )

            return success
        except Exception as e:
            self.handle_error(
                f"Error processing document {doc_url}",
                exception=e
            )
            return False

    def run_task(self) -> bool:
        if not self.check_doc():
            self.print_and_log_warning(
                "[task.extract_regions] No documents to annotate"
            )
            self.task_update(
                "ERROR",
                f"[API ERROR] Failed to download documents for {self.documents}",
            )
            return False

        self.task_update("STARTED")
        self.print_and_log(
            f"[task.extract_regions] Extraction task triggered with {self.model}!"
        )

        try:
            all_successful = True
            for doc_id, dataset_url in self.documents.items():
                success = self.process_doc(doc_id, dataset_url)
                all_successful = all_successful and success

            status = "SUCCESS" if all_successful else "ERROR"
            self.print_and_log(f"[task.extract_regions] Task completed with status: {status}")
            self.task_update(status, self.error_list if self.error_list else None)
            return all_successful
        except Exception as e:
            self.handle_error(str(e))
            self.task_update("ERROR", self.error_list)
            return False
