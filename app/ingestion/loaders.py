"""Document loaders: PDF (with OCR fallback), DOCX, TXT/MD and images (OCR)."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.config import get_settings

log = logging.getLogger(__name__)
logging.getLogger("pypdf").setLevel(logging.ERROR)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".png", ".jpg", ".jpeg", ".tiff"}


@dataclass
class Page:
    """A logical page. For DOCX/TXT this is a section or a block of text (they have no fixed pages)."""
    number: int
    text: str
    heading: Optional[str] = None
    ocr: bool = False


def load_document(path: Path) -> list[Page]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _load_pdf(path)
    if ext == ".docx":
        return _load_docx(path)
    if ext in {".txt", ".md"}:
        return _paginate_text(path.read_text(encoding="utf-8", errors="ignore"))
    if ext in {".png", ".jpg", ".jpeg", ".tiff"}:
        text = _ocr_image(path)
        return [Page(1, text, ocr=True)] if text else []
    raise ValueError(f"Unsupported file type: {ext}")


# ----------------------------------------------------------------------------- PDF
def _load_pdf(path: Path) -> list[Page]:
    from pypdf import PdfReader

    s = get_settings()
    reader = PdfReader(str(path))
    pages: list[Page] = []
    for i, p in enumerate(reader.pages, start=1):
        text = (p.extract_text() or "").strip()
        used_ocr = False
        if len(text) < s.ocr_min_chars and s.ocr_enabled:   # scanned page -> OCR
            ocr_text = _ocr_pdf_page(path, i)
            if ocr_text:
                text, used_ocr = ocr_text, True
        pages.append(Page(i, text, ocr=used_ocr))
    return pages


def _ocr_pdf_page(path: Path, page_number: int) -> str:
    try:
        import pytesseract
        from pdf2image import convert_from_path
    except ImportError:
        log.warning("OCR libraries not installed; skipping OCR for %s p.%s", path.name, page_number)
        return ""
    try:
        images = convert_from_path(str(path), first_page=page_number, last_page=page_number, dpi=200)
        return "\n".join(pytesseract.image_to_string(im) for im in images).strip()
    except Exception as exc:  # tesseract/poppler missing, corrupt page, ...
        log.warning("OCR failed for %s p.%s: %s", path.name, page_number, exc)
        return ""


def _ocr_image(path: Path) -> str:
    try:
        import pytesseract
        from PIL import Image
        return pytesseract.image_to_string(Image.open(path)).strip()
    except Exception as exc:
        log.warning("Image OCR failed for %s: %s", path.name, exc)
        return ""


# ----------------------------------------------------------------------------- DOCX
def _load_docx(path: Path) -> list[Page]:
    """Split a DOCX into sections at Heading 1/2 so citations can point at a section."""
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = docx.Document(str(path))
    sections: list[tuple[Optional[str], list[str]]] = [(None, [])]
    for block in document.iter_inner_content():
        if isinstance(block, Paragraph):
            text = block.text.strip()
            if not text:
                continue
            style = (block.style.name if block.style is not None else "") or ""
            if re.match(r"(Heading [12]|Title)$", style):
                sections.append((text, [text]))
            else:
                sections[-1][1].append(text)
        elif isinstance(block, Table):
            for row in block.rows:
                cells = [c.text.strip() for c in row.cells]
                sections[-1][1].append(" | ".join(c for c in cells if c))
    pages = []
    for heading, lines in sections:
        body = "\n\n".join(lines).strip()
        if body:
            pages.append(Page(len(pages) + 1, body, heading=heading))
    return pages


# ----------------------------------------------------------------------------- TXT
def _paginate_text(text: str, target: int = 2500) -> list[Page]:
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    pages, cur, size = [], [], 0
    for b in blocks:
        if cur and size + len(b) > target:
            pages.append(Page(len(pages) + 1, "\n\n".join(cur)))
            cur, size = [], 0
        cur.append(b)
        size += len(b)
    if cur:
        pages.append(Page(len(pages) + 1, "\n\n".join(cur)))
    return pages
