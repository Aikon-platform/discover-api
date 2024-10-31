import os
import platform
import sys
from pathlib import Path
from typing import Optional
import requests
import torch

from ..const import DEFAULT_MODEL, ANNO_PATH, MODEL_PATH, IMG_PATH
from ...shared.tasks import LoggedTask, Task
from ...shared.utils.fileutils import empty_file
from ...shared.utils.download import download_dataset

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from .yolov5.models.common import DetectMultiBackend
from .yolov5.utils.dataloaders import (
    IMG_FORMATS,
    VID_FORMATS,
    LoadImages,
    LoadScreenshots,
    LoadStreams,
)
from .yolov5.utils.general import (
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    cv2,
    increment_path,
    non_max_suppression,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from .yolov5.utils.torch_utils import select_device, smart_inference_mode
from ...shared.utils.img import get_img_paths

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = "api" / Path(os.path.relpath(ROOT, Path.cwd()))  # relative


@smart_inference_mode()
def detect(
    weights=ROOT / "yolov5s.pt",  # model path or triton URL
    source=ROOT / "data/images",  # file/dir/URL/glob/screen/0(webcam)
    data=ROOT / "data/coco128.yaml",  # dataset.yaml path
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_txt=False,  # save results to *.txt
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project=ROOT / "runs/detect",  # save results to project/name
    name="exp",  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    vid_stride=1,  # video frame-rate stride
    anno_file="annotation.txt",  # annotation in which write the output
    img_nb=1,  # increment of the image
):
    source = str(source)
    save_img = not nosave and not source.endswith(".txt")  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".txt") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(
            source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride
        )
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(
            source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride
        )
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for i, (path, im, im0s, vid_cap, s) in enumerate(dataset, start=1):
        with open(anno_file, "a") as f:
            f.write(f"{img_nb} {path.split('/')[-1]}\n")  # write canvas and image name
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = (
                increment_path(save_dir / Path(path).stem, mkdir=True)
                if visualize
                else False
            )
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(
                pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det
            )

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f"{i}: "
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / "labels" / p.stem) + (
                "" if dataset.mode == "image" else f"_{frame}"
            )  # im.txt
            s += "%gx%g " % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    x, y, width, height = (
                        int(xyxy[0]),
                        int(xyxy[1]),
                        int(xyxy[2] - xyxy[0]),
                        int(xyxy[3] - xyxy[1]),
                    )
                    with open(anno_file, "a") as f:
                        f.write(
                            f"{x} {y} {width} {height}\n"
                        )  # write image coordinates
                    if save_txt:  # Write to file
                        xywh = (
                            (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn)
                            .view(-1)
                            .tolist()
                        )  # normalized xywh
                        line = (
                            (cls, *xywh, conf) if save_conf else (cls, *xywh)
                        )  # label format
                        with open(f"{txt_path}.txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = (
                            None
                            if hide_labels
                            else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                        )
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(
                            xyxy,
                            imc,
                            file=save_dir / "crops" / names[c] / f"{p.stem}.jpg",
                            BGR=True,
                        )

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == "Linux" and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(
                        str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO
                    )  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == "image":
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(
                            Path(save_path).with_suffix(".mp4")
                        )  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
                        )
                    vid_writer[i].write(im0)
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


class ExtractRegions(Task):
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

    def check_doc(self):
        # TODO check URL in document list
        if not self.documents:
            return False
        return True

    def send_annotations(
        self, experiment_id, annotation_file, dataset_ref
    ):
        if not self.notify_url:
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


class LoggedExtractRegions(LoggedTask, ExtractRegions):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process_img(
        self,
        img_path: Path,
        annotation_file: Path,
        img_number: int
    ) -> bool:
        filename = img_path.name
        try:
            self.print_and_log(f"====> Processing {filename} 🔍")
            detect(
                weights=self.weights,
                source=img_path,
                anno_file=str(annotation_file),
                img_nb=img_number,
            )
        except Exception as e:
            self.handle_error(
                f"Error processing image {filename}",
                exception=e
            )
            return False
        return True

    def process_doc_imgs(
        self,
        dataset_path: Path,
        annotation_file: Path
    ) -> bool:
        images = get_img_paths(dataset_path)
        try:
            for i, image in enumerate(images, 1):
                self.process_img(image, annotation_file, i)
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
            self.print_and_log(f"[task.extract_objects] Downloading {doc_id}...")

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

            # Process images
            self.print_and_log(f"DETECTING VISUAL ELEMENTS FOR {doc_url} 🕵️")
            dataset_path = IMG_PATH / image_dir
            self.process_doc_imgs(dataset_path, annotation_file)

            # Send annotations
            success = self.send_doc_regions(
                doc_url,
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

    def send_doc_regions(
        self,
        document: str,
        annotation_file: Path,
        dataset_ref: str,
    ) -> bool:
        """Send annotations for a document"""
        try:
            self.send_annotations(
                self.experiment_id,
                annotation_file,
                dataset_ref,
            )
            self.print_and_log(
                f"[task.regions] Successfully sent annotation for {document}"
            )
            return True
        except Exception as e:
            self.handle_error(
                f"Failed to send annotation for {document}",
                exception=e
            )
            return False

    def run_task(self) -> bool:
        """Main task execution method"""
        if not self.check_doc():
            self.print_and_log_warning(
                "[task.extract_objects] No documents to annotate"
            )
            self.task_update(
                "ERROR",
                f"[API ERROR] Failed to download documents for {self.documents}",
            )
            return False

        try:
            self.print_and_log(
                f"[task.extract_objects] Extraction task triggered with {self.model}!"
            )
            self.task_update("STARTED")

            all_successful = True
            for doc_id, dataset_url in self.documents.items():
                success = self.process_doc(doc_id, dataset_url)
                all_successful = all_successful and success

            status = "SUCCESS" if all_successful else "ERROR"
            self.task_update(status, self.error_list if self.error_list else None)
            return all_successful

        except Exception as e:
            self.handle_error(str(e))
            self.task_update("ERROR", self.error_list)
            return False
