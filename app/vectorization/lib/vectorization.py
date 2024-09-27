import re
import torch
import glob
import svgwrite

import xml.etree.ElementTree as ET
from typing import Optional

from .HDV.src.main import build_model_main
from .HDV.src.util.slconfig import SLConfig
from .HDV.src.util.visualizer import COCOVisualizer
from .HDV.src.datasets import transforms as T

from .utils import *
from pathlib import Path

from svg.path import parse_path
from svg.path.path import Line, Move, Arc

from ..const import IMG_PATH, MODEL_PATH, VEC_RESULTS_PATH, DEFAULT_EPOCHS
from ...shared.utils.fileutils import send_update
from ...shared.utils.img import download_img
from ...shared.utils.logging import LoggingTaskMixin
from ..lib.utils import is_downloaded


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
                "ERROR", f"[API ERROR] Failed to download documents for vectorization"
            )
            return

        error_list = []

        try:
            for doc_id, document in self.documents.items():
                self.print_and_log(
                    f"[task.vectorization] Vectorization task triggered for {doc_id} !"
                )
                self.task_update("STARTED")

                self.download_dataset(doc_id, document)
                self.process_inference(doc_id)
                self.send_zip(doc_id)

            self.task_update("SUCCESS", error_list if error_list else None)

        except Exception as e:
            self.print_and_log(f"Error when computing vectorizations", e=e)
            self.task_update("ERROR", f"[API ERROR] Vectorization task failed: {e}")

    def download_dataset(self, doc_id, document):
        self.print_and_log(f"[task.vectorization] Dowloading images...", color="blue")
        for image_id, url in document.items():
            try:
                if not is_downloaded(doc_id, image_id):
                    self.print_and_log(
                        f"[task.vectorization] Downloading image {image_id}"
                    )
                    download_img(url, doc_id, image_id, IMG_PATH, MAX_SIZE)

            except Exception as e:
                self.print_and_log(
                    f"[task.vectorization] Unable to download image {image_id}", e
                )

    def process_inference(self, doc_id):
        model_folder = Path(MODEL_PATH)
        model_config_path = f"{model_folder}/config_cfg.py"
        epoch = DEFAULT_EPOCHS if self.model is None else self.model
        model_checkpoint_path = f"{model_folder}/checkpoint{epoch}.pth"
        args = SLConfig.fromfile(model_config_path)
        args.device = "cuda"
        args.num_select = 200

        corpus_folder = Path(IMG_PATH)
        image_paths = glob.glob(str(corpus_folder / doc_id) + "/*.jpg")
        output_dir = VEC_RESULTS_PATH / doc_id
        os.makedirs(output_dir, exist_ok=True)

        model, criterion, postprocessors = build_model_main(args)
        checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        model.eval()

        args.dataset_file = "synthetic"
        args.mode = "primitives"
        args.relative = False
        args.common_queries = True
        args.eval = True
        args.coco_path = "data/synthetic_processed"
        args.fix_size = False
        args.batch_size = 1
        args.boxes_only = False
        vslzr = COCOVisualizer()
        id2name = {0: "line", 1: "circle", 2: "arc"}
        primitives_to_show = ["line", "circle", "arc"]

        torch.cuda.empty_cache()
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        with torch.no_grad():
            for image_path in image_paths:
                try:
                    self.print_and_log(
                        f"[task.vectorization] Processing {image_path}", color="blue"
                    )
                    # Load and process image
                    im_name = os.path.basename(image_path)[:-4]
                    image = Image.open(image_path).convert("RGB")
                    im_shape = image.size
                    input_image, _ = transform(image, None)
                    size = torch.Tensor([input_image.shape[1], input_image.shape[2]])

                    # Model inference
                    output = model.cuda()(input_image[None].cuda())
                    output = postprocessors["param"](
                        output,
                        torch.Tensor([[im_shape[1], im_shape[0]]]).cuda(),
                        to_xyxy=False,
                    )[0]

                    threshold, arc_threshold = 0.3, 0.3
                    scores = output["scores"]
                    labels = output["labels"]
                    boxes = output["parameters"]
                    select_mask = ((scores > threshold) & (labels != 2)) | (
                        (scores > arc_threshold) & (labels == 2)
                    )
                    labels = labels[select_mask]
                    boxes = boxes[select_mask]
                    scores = scores[select_mask]
                    pred_dict = {
                        "parameters": boxes,
                        "labels": labels,
                        "scores": scores,
                    }
                    (
                        lines,
                        line_scores,
                        circles,
                        circle_scores,
                        arcs,
                        arc_scores,
                    ) = get_outputs_per_class(pred_dict)

                    # Postprocess the outputs
                    lines, line_scores = remove_duplicate_lines(
                        lines, im_shape, line_scores
                    )
                    lines, line_scores = remove_small_lines(
                        lines, im_shape, line_scores
                    )
                    circles, circle_scores = remove_duplicate_circles(
                        circles, im_shape, circle_scores
                    )
                    arcs, arc_scores = remove_duplicate_arcs(arcs, im_shape, arc_scores)
                    arcs, arc_scores = remove_arcs_on_top_of_circles(
                        arcs, circles, im_shape, arc_scores
                    )
                    arcs, arc_scores = remove_arcs_on_top_of_lines(
                        arcs, lines, im_shape, arc_scores
                    )

                    # Generate and save SVG
                    self.print_and_log(
                        f"[task.vectorization] Drawing {image_path}", color="blue"
                    )
                    # shutil.copy2(image_path, output_dir)
                    # décommenter cette ligne si on veut obtenir les images dans le répertoire de sortie
                    diagram_name = Path(image_path).stem
                    image_name = os.path.basename(image_path)
                    lines = lines.reshape(-1, 2, 2)
                    arcs = arcs.reshape(-1, 3, 2)

                    dwg = svgwrite.Drawing(
                        str(output_dir / f"{diagram_name}.svg"),
                        profile="tiny",
                        size=im_shape,
                    )
                    dwg.add(dwg.image(href=image_name, insert=(0, 0), size=im_shape))
                    dwg = write_svg_dwg(
                        dwg, lines, circles, arcs, show_image=False, image=None
                    )
                    dwg.save(pretty=True)

                    ET.register_namespace("", "http://www.w3.org/2000/svg")
                    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
                    ET.register_namespace(
                        "sodipodi", "http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
                    )
                    ET.register_namespace(
                        "inkscape", "http://www.inkscape.org/namespaces/inkscape"
                    )

                    file_name = output_dir / f"{diagram_name}.svg"
                    tree = ET.parse(file_name)
                    root = tree.getroot()

                    root.set(
                        "xmlns:inkscape", "http://www.inkscape.org/namespaces/inkscape"
                    )
                    root.set(
                        "xmlns:sodipodi",
                        "http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd",
                    )
                    root.set("inkscape:version", "1.3 (0e150ed, 2023-07-21)")

                    arc_regex = re.compile(r"[aA]")
                    for path in root.findall("{http://www.w3.org/2000/svg}path"):
                        d = path.get("d", "")
                        if arc_regex.search(d):
                            path.set("sodipodi:type", "arc")
                            path.set("sodipodi:arc-type", "arc")
                            path_parsed = parse_path(d)
                            for e in path_parsed:
                                if isinstance(e, Line):
                                    continue
                                elif isinstance(e, Arc):
                                    (
                                        center,
                                        radius,
                                        start_angle,
                                        end_angle,
                                        p0,
                                        p1,
                                    ) = get_arc_param([e])
                                    path.set("sodipodi:cx", f"{center[0]}")
                                    path.set("sodipodi:cy", f"{center[1]}")
                                    path.set("sodipodi:rx", f"{radius}")
                                    path.set("sodipodi:ry", f"{radius}")
                                    path.set("sodipodi:start", f"{start_angle}")
                                    path.set("sodipodi:end", f"{end_angle}")

                    tree.write(file_name, xml_declaration=True)

                    self.print_and_log(
                        f"[task.vectorization] SVG for {image_path} drawn",
                        color="yellow",
                    )

                except Exception as e:
                    self.print_and_log(
                        f"[task.vectorization] Failed to process {image_path}", e
                    )

            self.print_and_log(f"[task.vectorization] Task over", color="yellow")

    def send_zip(self, doc_id):
        """
        Zip le répertoire correspondant à doc_id et envoie ce répertoire via POST à l'URL spécifiée.
        """
        try:
            output_dir = VEC_RESULTS_PATH / doc_id
            zip_path = output_dir / f"{doc_id}.zip"
            self.print_and_log(
                f"[task.vectorization] Zipping directory {output_dir}", color="blue"
            )

            zip_directory(output_dir, zip_path)
            self.print_and_log(
                f"[task.vectorization] Sending zip {zip_path} to {self.notify_url}",
                color="blue",
            )

            with open(zip_path, "rb") as zip_file:
                response = requests.post(
                    url=self.notify_url,
                    files={
                        "file": zip_file,
                    },
                    data={
                        "experiment_id": self.experiment_id,
                        "model": self.model,
                    },
                )

            if response.status_code == 200:
                self.print_and_log(
                    f"[task.vectorization] Zip sent successfully to {self.notify_url}",
                    color="yellow",
                )
            else:
                self.print_and_log(
                    f"[task.vectorization] Failed to send zip to {self.notify_url}. Status code: {response.status_code}",
                    color="red",
                )

        except Exception as e:
            self.print_and_log(
                f"[task.vectorization] Failed to zip and send directory {output_dir}", e
            )
