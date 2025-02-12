import gc
import json
import os
from pathlib import Path
from typing import Optional

from .const import DEFAULT_MODEL, MODEL_PATH
from .lib.extract import YOLOExtractor, FasterRCNNExtractor
from ..shared.tasks import LoggedTask
from ..shared.dataset import Document, Dataset, Image as DImage
from ..shared.utils.fileutils import get_model

EXTRACTOR_POSTPROCESS_KWARGS = {
    "watermarks": {
        "squarify": True,
        "margin": 0.05,
    },
}


class ExtractRegions(LoggedTask):
    """
    Task to extract regions from a dataset

    Args:
        dataset (Dataset): The dataset to process
        model (str, optional): The model file name stem to use for extraction (default: DEFAULT_MODEL)
    """

    def __init__(
        self,
        dataset: Dataset,
        model: Optional[str] = None,
        postprocess: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dataset = dataset
        self._model = model
        self._extraction_model: Optional[str] = None

        self.result_dir = Path()
        self.result_urls = []
        self.annotations = {}
        self.extractor = None
        print("POSTPROCESS", postprocess)
        self.extractor_kwargs = EXTRACTOR_POSTPROCESS_KWARGS.get(postprocess, {})

    def initialize(self):
        """
        Initialize the extractor, based on the model's name prefix
        """
        if self.model.startswith(("rcnn", "fasterrcnn")):
            self.extractor = FasterRCNNExtractor(self.weights, **self.extractor_kwargs)
        else:
            self.extractor = YOLOExtractor(self.weights, **self.extractor_kwargs)

    def terminate(self):
        """
        Clear memory
        """
        self.annotations = {}
        self.extractor = None
        gc.collect()

    @property
    def model(self) -> str:
        return DEFAULT_MODEL if self._model is None else self._model

    @property
    def weights(self) -> Path:
        # return MODEL_PATH / self.model
        return get_model(self.model, MODEL_PATH)

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

    def process_img(self, img: DImage, extraction_ref: str, doc_uid: str) -> bool:
        """
        Process a single image, appends the annotations to self.annotations[extraction_ref]
        """
        try:
            self.print_and_log(f"====> Processing {img.path.name} 🔍")
            anno = self.extractor.extract_one(img)
            anno["doc_uid"] = doc_uid
            self.annotations[extraction_ref].append(anno)
            return True
        except Exception as e:
            self.handle_error(f"Error processing image {img.path.name}", exception=e)
            return False

    def process_doc_imgs(self, doc: Document, extraction_ref: str) -> bool:
        """
        Process all images in a document, store the annotations in self.annotations[extraction_ref] (clears it first)
        """
        images = doc.list_images()
        self.annotations[extraction_ref] = []
        try:
            for i, image in enumerate(
                self.jlogger.iterate(images, "Analyzing images"), 1
            ):
                success = self.process_img(image, extraction_ref, doc.uid)
                if not success:
                    self.handle_error(f"Failed to process {image}")
        except Exception as e:
            self.handle_error(f"Error processing images for {doc.uid}", exception=e)
            return False
        return True

    def process_doc(self, doc: Document) -> bool:
        """
        Process a whole document, download it, process all images, save annotations
        """
        try:
            self.print_and_log(f"[task.extract_regions] Downloading {doc.uid}...")

            doc.download()
            if not doc.has_images():
                self.handle_error(f"No images were extracted from {doc.uid}")
                return False

            self.result_dir = doc.annotations_path
            os.makedirs(self.result_dir, exist_ok=True)

            # This way, same dataset can be extracted twice with same extraction model
            # is it what we want?
            extraction_ref = f"{self.extraction_model}+{self.experiment_id}"
            annotation_file = self.result_dir / f"{extraction_ref}.json"
            with open(annotation_file, "w"):
                pass

            extraction_id = f"{doc.uid}@@{extraction_ref}"

            self.print_and_log(f"DETECTING VISUAL ELEMENTS FOR {doc.uid} 🕵️")
            success = self.process_doc_imgs(doc, extraction_id)
            if success:
                with open(annotation_file, "w") as f:
                    json.dump(self.annotations[extraction_id], f, indent=2)
                result_url = doc.get_annotations_url(extraction_ref)
                self.notifier(
                    "PROGRESS", output={"annotations": [{doc.uid: result_url}]}
                )
                self.result_urls.append({doc.uid: result_url})

            return success
        except Exception as e:
            self.handle_error(f"Error processing document {doc.uid}", exception=e)
            return False

    def run_task(self) -> bool:
        """
        Run the extraction task
        """
        if not self.check_doc():
            self.print_and_log_warning("[task.extract_regions] No dataset to annotate")
            self.task_update(
                "ERROR",
                message=f"[API ERROR] Failed to download dataset for {self.dataset}",
            )
            return False

        self.task_update("STARTED")
        self.print_and_log(
            f"[task.extract_regions] Extraction task triggered with {self.model}!"
        )

        try:
            self.initialize()
            all_successful = True
            for doc in self.jlogger.iterate(
                self.dataset.documents, "Processing documents"
            ):
                success = self.process_doc(doc)
                all_successful = all_successful and success

            status = "SUCCESS" if all_successful else "ERROR"
            self.print_and_log(
                f"[task.extract_regions] Task completed with status: {status}"
            )
            self.task_update(status, message=self.error_list if self.error_list else [])
            return all_successful
        except Exception as e:
            self.handle_error(f"Error {e} processing dataset", exception=e)
            self.task_update("ERROR", message=self.error_list)
            return False
        finally:
            self.terminate()
