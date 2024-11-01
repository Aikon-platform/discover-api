import os
import sys
from pathlib import Path
import torch

from ultralytics.utils.plotting import Annotator, colors

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
CONF_THRES = 0.25  # confidence threshold
IOU_THRES = 0.45  # IOU threshold for non-maximum suppression
HIDE_LABEL = False  # hide/show labels if save_img
HIDE_CONF = False  # hide/show confidence if save_img


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


def write_annotation(anno_file, img_nb, path, x, y, width, height):
    with open(anno_file, "a") as f:
        if x is None:  # Write image name
            f.write(f"{img_nb} {path.split('/')[-1]}\n")
        else:  # Write detection
            f.write(f"{x} {y} {width} {height}\n")


def save_extraction_img(source, annotator):
    output_path = str(Path(source).parent / f"extracted_{Path(source).name}")
    cv2.imwrite(output_path, annotator.result())


def process_detections(det, im, im0s, names, save_img, source, anno_file):
    annotator = Annotator(im0s, line_width=2, example=str(names)) if save_img else None

    if len(det):
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0s.shape).round()

        # Write results
        for *xyxy, conf, cls in reversed(det):
            # Extract coordinates
            x, y, width, height = (
                int(xyxy[0]),
                int(xyxy[1]),
                int(xyxy[2] - xyxy[0]),
                int(xyxy[3] - xyxy[1]),
            )

            # Write coordinates to annotation file
            write_annotation(anno_file, None, None, x, y, width, height)

            if save_img:  # Add bbox to image if saving
                c = int(cls)  # integer class
                label = (
                    None
                    if HIDE_LABEL
                    else (names[c] if HIDE_CONF else f"{names[c]} {conf:.2f}")
                )
                annotator.box_label(xyxy, label, color=colors(c, True))

        if save_img:
            save_extraction_img(source, annotator)


@smart_inference_mode()
def extract(
    weights=ROOT / "yolov5s.pt",  # model path
    source=ROOT / "data/images",  # path of image to be processed
    anno_file="annotation.txt",  # annotation output file
    img_nb=1,  # image number for annotation
    imgsz=(640, 640),  # inference size (height, width)
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    save_img=False,  # save inference images
    classes=None,  # filter by class: --class 0, or --class 0 2 3
):
    source = setup_source(source)
    model = DetectMultiBackend(weights, device=select_device(device), fp16=False)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)

    model.warmup(imgsz=(1 if pt or model.triton else 1, 3, *imgsz))
    for i, (path, im, im0s, vid_cap, s) in enumerate(dataset, start=1):
        write_annotation(anno_file, img_nb, path, None, None, None, None)

        im = prepare_image(im, model.device)
        pred = model(im, augment=False)
        pred = non_max_suppression(pred, CONF_THRES, IOU_THRES, classes, False, max_det=1000)

        for det in pred:
            process_detections(det, im, im0s, names, save_img, source, anno_file)
