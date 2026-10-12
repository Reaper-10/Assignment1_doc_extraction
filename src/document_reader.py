import os
from pathlib import Path

from docx import Document
from PIL import Image, ImageOps, ImageEnhance
import pytesseract


def resolve_tesseract_executable() -> str | None:
    """Find the installed Tesseract executable on common Windows paths."""
    env_path = os.getenv("TESSERACT_CMD") or os.getenv("TESSERACT_PATH")
    if env_path and Path(env_path).exists():
        return env_path

    candidates = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        str(Path.home() / "AppData" / "Local" / "Programs" / "Tesseract-OCR" / "tesseract.exe"),
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    return None


def _configure_tesseract() -> None:
    resolved = resolve_tesseract_executable()
    if resolved:
        pytesseract.pytesseract.tesseract_cmd = resolved


def read_docx(path: str | Path) -> str:
    """Extract text from paragraphs and tables, preserving table row context."""
    doc = Document(str(path))
    chunks = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    for table_idx, table in enumerate(doc.tables, start=1):
        chunks.append(f"[TABLE {table_idx}]")
        for row in table.rows:
            cells = [cell.text.strip().replace("\n", " / ") for cell in row.cells]
            if any(cells):
                chunks.append(" | ".join(cells))
    return "\n".join(chunks).strip()


def read_image(path: str | Path) -> str:
    """OCR a scanned image; Tesseract must be installed separately."""
    _configure_tesseract()
    image = Image.open(path).convert("L")
    image = ImageOps.autocontrast(image)
    image = ImageEnhance.Contrast(image).enhance(1.4)
    # OCR is text acquisition only; semantic field extraction is done by the ML model.
    return pytesseract.image_to_string(image, config="--psm 3").strip()


def read_document(path: str | Path) -> str:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".docx":
        return read_docx(path)
    if suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
        return read_image(path)
    raise ValueError(f"Unsupported file type: {suffix}. Supported: DOCX and common image formats.")
