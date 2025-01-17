from pathlib import Path
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..utils.logging import console

if TYPE_CHECKING:
    from .document import Document


@dataclass
class Image:
    id: str
    src: str
    path: Path
    document: "Document" = None


def pdf_to_img(pdf_path, img_path, dpi=500, max_size=3000):
    """
    Convert the PDF file to JPEG images
    """
    import subprocess

    file_prefix = pdf_path.stem
    try:
        command = f"pdftoppm -jpeg -r {dpi} -scale-to {max_size} {pdf_path} {img_path}/{file_prefix} -sep _ "
        subprocess.run(command, shell=True, check=True)

    except Exception as e:
        console(f"Error converting {pdf_path} to images: {e}", "red")
