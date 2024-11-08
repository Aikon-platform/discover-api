import os
import sys
from pathlib import Path
import torch
from PIL import Image

from ultralytics.utils.plotting import Annotator, colors
from .bbox import Segment  # Import the Segment class

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


def get_img_dim(source):
    with Image.open(source) as img:
        return img.size[0], img.size[1]  # width, height


class AnnotationWriter:
    def __init__(self, txt_file):
        self.txt_file = txt_file
        self.current_image = None
        self.annotations = []
        self.current_crops = []

    def add_img(self, img_nb, path, img_w=None, img_h=None):
        if self.current_image:
            self.annotations.append(self.current_image)
            self.current_image = None

        if img_w is None or img_h is None:
            img_w, img_h = get_img_dim(path)

        self.current_image = {
            "source": Path(path).name,
            "nb": img_nb,
            "width": img_w,
            "height": img_h,
            "crops": []
        }

        with open(self.txt_file, "a") as f:
            f.write(f"{img_nb} {path.split('/')[-1]}\n")

    def add_region(self, x, y, w, h):
        img_w = self.current_image["width"]
        img_h = self.current_image["height"]

        rel_x = x / img_w
        rel_y = y / img_h
        rel_w = w / img_w
        rel_h = h / img_h

        segment = Segment(rel_x, rel_y, rel_w, rel_h, precision=2)

        x1, y1 = x, y
        x2, y2 = x + w, y + h

        self.current_image["crops"].append({
            "bbox": segment.serialize(),  # compact string representation
            "absolute": {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "width": w,
                "height": h
            },
            "relative": {
                "x1": round(rel_x, 4),
                "y1": round(rel_y, 4),
                "x2": round(rel_x + rel_w, 4),
                "y2": round(rel_y + rel_h, 4),
                "width": round(rel_w, 4),
                "height": round(rel_h, 4)
            }
        })

        with open(self.txt_file, "a") as f:
            f.write(f"{x} {y} {w} {h}\n")

    def save(self):
        if self.current_image:
            self.annotations.append(self.current_image)


def process_detections(det, im, im0s, names, save_img, source, writer):
    annotator = Annotator(im0s, line_width=2, example=str(names)) if save_img else None

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

            writer.add_region(x, y, w, h)

            if save_img:
                c = int(cls)
                label = None if HIDE_LABEL else (names[c] if HIDE_CONF else f"{names[c]} {conf:.2f}")
                annotator.box_label(xyxy, label, color=colors(c, True))

        if save_img:
            output_path = str(Path(source).parent / f"extracted_{Path(source).name}")
            cv2.imwrite(output_path, annotator.result())


def setup_source(source):
    source = str(source)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    is_file = Path(source).suffix[1:] in IMG_FORMATS
    if is_url and is_file:
        source = check_file(source)  # download
    return source


def prepare_image(im, device):
    im = torch.from_numpy(im).to(device)
    im = im.float()  # uint8 to fp32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    return im


@smart_inference_mode()
def extract(
    weights=ROOT / "yolov5s.pt",
    source=ROOT / "data/images",  # path of the image to be processed
    anno_file="annotation.txt",
    img_nb=1,
    imgsz=(640, 640),
    device="",
    save_img=False,
    classes=None,
):
    source = setup_source(source)

    model = DetectMultiBackend(weights, device=select_device(device), fp16=False)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    writer = AnnotationWriter(anno_file)
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)

    model.warmup(imgsz=(1 if pt or model.triton else 1, 3, *imgsz))
    for i, (path, im, im0s, vid_cap, s) in enumerate(dataset, start=1):
        writer.add_img(img_nb, path, img_w=im0s.shape[1], img_h=im0s.shape[0])

        im = prepare_image(im, model.device)
        pred = model(im, augment=False)
        pred = non_max_suppression(pred, CONF_THRES, IOU_THRES, classes, False, max_det=1000)

        for det in pred:
            process_detections(det, im, im0s, names, save_img, source, writer)

    writer.save()
    return writer.annotations
