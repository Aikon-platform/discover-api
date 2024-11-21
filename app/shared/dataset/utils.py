from pathlib import Path
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .document import Document

@dataclass
class Image:
    id: str
    src: str
    path: Path
    document: "Document"=None
