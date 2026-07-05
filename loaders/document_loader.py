from __future__ import annotations

from pathlib import Path
from typing import Union


SUPPORTED_DOCUMENT_EXTENSIONS = {".pdf", ".txt", ".docx"}


def save_uploaded_file(uploaded_file, upload_dir: Path) -> Path:
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(uploaded_file.name).name
    target_path = upload_dir / safe_name
    target_path.write_bytes(uploaded_file.getbuffer())
    return target_path


def load_document_text(path: Union[str, Path]) -> str:
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return _read_pdf(file_path)
    if suffix == ".txt":
        return file_path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".docx":
        return _read_docx(file_path)

    supported = ", ".join(sorted(SUPPORTED_DOCUMENT_EXTENSIONS))
    raise ValueError(f"Unsupported document type `{suffix}`. Supported types: {supported}.")


def _read_pdf(file_path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(file_path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n\n".join(pages)


def _read_docx(file_path: Path) -> str:
    from docx import Document

    document = Document(str(file_path))
    return "\n".join(paragraph.text for paragraph in document.paragraphs)
