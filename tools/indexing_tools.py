from __future__ import annotations

from core.config import PROJECT_ROOT
from core.index_history import add_index_history_record
from loaders.code_loader import load_code_project
from loaders.document_loader import load_document_text, save_uploaded_file
from rag.chunking import split_text
from rag.vector_store import VectorStore


UPLOAD_DIR = PROJECT_ROOT / "data" / "uploaded_files"


def add_text_to_index(
    vector_store: VectorStore,
    text: str,
    source: str,
    document_id: str,
    kind: str,
) -> int:
    chunks = split_text(text)
    if not chunks:
        return 0

    vector_store.add_chunks(
        chunks,
        source=source,
        document_id=document_id,
        kind=kind,
    )
    return len(chunks)


def index_uploaded_files(uploaded_files, vector_store: VectorStore) -> int:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    total_chunks = 0
    for uploaded_file in uploaded_files:
        saved_path = save_uploaded_file(uploaded_file, UPLOAD_DIR)
        text = load_document_text(saved_path)
        chunk_count = add_text_to_index(
            vector_store,
            text,
            source=saved_path.name,
            document_id=f"document:{saved_path.name}",
            kind="document",
        )
        if chunk_count == 0:
            continue

        total_chunks += chunk_count
        add_index_history_record(
            source_type="document",
            source=saved_path.name,
            chunks=chunk_count,
        )

    return total_chunks


def index_code_folder(folder_path: str, vector_store: VectorStore) -> int:
    files = load_code_project(folder_path)
    total_chunks = 0

    for file_info in files:
        text = f"File: {file_info['source']}\n\n{file_info['text']}"
        total_chunks += add_text_to_index(
            vector_store,
            text,
            source=file_info["source"],
            document_id=f"code:{file_info['source']}",
            kind="code",
        )

    add_index_history_record(
        source_type="code folder",
        source=folder_path,
        chunks=total_chunks,
        message=f"{len(files)} files read",
    )
    return total_chunks
