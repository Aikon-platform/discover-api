import os
import sys
from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image
from typing import Tuple

from ultralytics.utils.plotting import Annotator, colors
from .bbox import Segment

from .yolov5.models.common import DetectMultiBackend
from .yolov5.utils.dataloaders import IMG_FORMATS, LoadImages
from .yolov5.utils.general import (
    check_file,
    check_img_size,
    cv2,
    non_max_suppression,
    scale_boxes,
)
from .yolov5.utils.torch_utils import select_device, smart_inference_mode

from ...shared.utils.fileutils import TPath

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = "api" / Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# Constants
CONF_THRES = 0.25
IOU_THRES = 0.45
HIDE_LABEL = False
HIDE_CONF = False

# UTILS


def get_img_dim(source: TPath) -> Tuple[int, int]:
    """
    Get the dimensions of an image (width, height)
    """
    with Image.open(source) as img:
        return img.size[0], img.size[1]  # width, height


def setup_source(source: TPath) -> str:
    """
    Check if the source is a URL or a file

    If the source is a URL that points to an image file, download the image
    """
    source = str(source)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    is_file = Path(source).suffix[1:] in IMG_FORMATS
    if is_url and is_file:
        source = check_file(source)  # download
    return source


class ImageAnnotator:
    def __init__(self, path: TPath, img_w: int = None, img_h: int = None):
        if img_w is None or img_h is None:
            img_w, img_h = get_img_dim(path)

        self.annotations = {
            "source": Path(path).name,
            "width": img_w,
            "height": img_h,
            "crops": [],
        }

    def add_region(self, x: int, y: int, w: int, h: int, conf: float):
        img_w = self.annotations["width"]
        img_h = self.annotations["height"]

        rel_x = x / img_w
        rel_y = y / img_h
        rel_w = w / img_w
        rel_h = h / img_h

        segment = Segment(rel_x, rel_y, rel_w, rel_h, precision=2)

        x1, y1 = x, y
        x2, y2 = x + w, y + h

        self.annotations["crops"].append(
            {
                "bbox": segment.serialize(),  # compact string representation
                "crop_id": f'{self.annotations["source"]}-{segment.serialize()}',
                "source": self.annotations["source"],
                "confidence": round(conf, 4),
                "absolute": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "width": w,
                    "height": h,
                },
                "relative": {
                    "x1": round(rel_x, 4),
                    "y1": round(rel_y, 4),
                    "x2": round(rel_x + rel_w, 4),
                    "y2": round(rel_y + rel_h, 4),
                    "width": round(rel_w, 4),
                    "height": round(rel_h, 4),
                },
            }
        )


class BaseExtractor:
    """
    A base class for extracting regions from images
    """

    DEFAULT_IMG_SIZE = (640, 640)

    def __init__(self, weights: TPath, device: str = "cpu", imgsz=None):
        self.weights = weights
        self.device = torch.device(device)
        self.imgsz = imgsz if imgsz is not None else self.DEFAULT_IMG_SIZE
        self.model = self.get_model()

    def get_model(self):
        raise NotImplementedError()

    @smart_inference_mode()
    def extract_one(self, img_path: TPath):
        raise NotImplementedError()

    @smart_inference_mode()
    def prepare_image(self, im):
        return transforms.ToTensor()(im).unsqueeze(0).to(self.device)


class YOLOExtractor(BaseExtractor):
    def get_model(self):
        self.device = select_device(self.device)
        return DetectMultiBackend(self.weights, device=self.device, fp16=False)

    def prepare_image(self, im):
        return (torch.from_numpy(im).to(self.device).float() / 255.0).unsqueeze(
            0
        )  # no need to swap axes

    @smart_inference_mode()
    def process_detections(
        self,
        det,
        im: torch.Tensor,
        im0s,
        names,
        save_img: bool,
        source: TPath,
        writer: ImageAnnotator,
    ):
        annotator = (
            Annotator(im0s, line_width=2, example=str(names)) if save_img else None
        )

        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0s.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                # Extract coordinates
                x, y, w, h = (
                    int(xyxy[0]),
                    int(xyxy[1]),
                    int(xyxy[2] - xyxy[0]),
                    int(xyxy[3] - xyxy[1]),
                )

                writer.add_region(x, y, w, h, float(conf))

                if save_img:
                    c = int(cls)
                    label = (
                        None
                        if HIDE_LABEL
                        else (names[c] if HIDE_CONF else f"{names[c]} {conf:.2f}")
                    )
                    annotator.box_label(xyxy, label, color=colors(c, True))

            if save_img:
                output_path = str(
                    Path(source).parent / f"extracted_{Path(source).name}"
                )
                cv2.imwrite(output_path, annotator.result())

    @smart_inference_mode()
    def extract_one(self, img_path, save_img: bool = False):
        source = setup_source(img_path)

        stride, names, pt = self.model.stride, self.model.names, self.model.pt
        imgsz = check_img_size(self.imgsz, s=stride)

        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)

        self.model.warmup(imgsz=(1 if pt or self.model.triton else 1, 3, *imgsz))
        assert len(dataset) == 1, f"Error: {len(dataset)} images found, expected 1"

        for i, (path, im, im0s, vid_cap, s) in enumerate(dataset, start=1):
            assert i == 1, "Error: Multiple images not supported"
            writer = ImageAnnotator(path, img_w=im0s.shape[1], img_h=im0s.shape[0])

            im = self.prepare_image(im)
            pred = self.model(im, augment=False)
            pred = non_max_suppression(
                pred, CONF_THRES, IOU_THRES, None, False, max_det=1000
            )

            for det in pred:
                self.process_detections(det, im, im0s, names, save_img, source, writer)

        return writer.annotations


class FasterRCNNExtractor(BaseExtractor):
    DEFAULT_IMG_SIZE = [800, 1400, 2000]  # used for multiscale inference

    def get_model(self):
        model = torch.load(self.weights, map_location=self.device).eval()
        return model

    @smart_inference_mode()
    def process_detections(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        im0s: Image.Image,
        im: torch.Tensor,
        writer: ImageAnnotator,
    ):
        boxes[:, :4] = scale_boxes(im.shape[2:], boxes[:, :4], im0s.size[::-1]).round()

        for box, score in zip(boxes, scores):
            if score < 0.3:
                break
            x1, y1, x2, y2 = box
            writer.add_region(
                int(x1), int(y1), int(x2 - x1), int(y2 - y1), float(score)
            )

    @smart_inference_mode()
    def extract_one(self, img_path: TPath):
        source = setup_source(img_path)

        im0s = Image.open(source).convert("RGB")
        writer = ImageAnnotator(source, img_w=im0s.size[0], img_h=im0s.size[1])

        for size in self.imgsz:
            im = im0s.copy()
            im.thumbnail((size, size))

            im = self.prepare_image(im)
            preds = self.model(im)

            boxes = preds[0]["boxes"].cpu().numpy()
            scores = preds[0]["scores"].cpu().numpy()

            if len(scores) == 0:
                continue

            if scores[0] > 0.4:
                break

        self.process_detections(boxes, scores, im0s, im, writer)
        return writer.annotations
