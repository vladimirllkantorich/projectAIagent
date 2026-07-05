from __future__ import annotations

import importlib
import sys

import streamlit as st

from core.config import load_config
from core.python_runtime import TARGET_PYTHON, python_runtime_status
from rag.embeddings import EmbeddingModel
from rag.vector_store import VectorStore


MODULES_TO_CHECK = [
    "streamlit",
    "crewai",
    "openai",
    "chromadb",
    "pypdf",
    "docx",
]


def check_python_runtime() -> bool:
    ok = True
    print("DevVault AI health check")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Target Python: {TARGET_PYTHON}")

    runtime_status, runtime_message = python_runtime_status()
    print(f"[{runtime_status}] {runtime_message}")
    if runtime_status == "error":
        ok = False
    return ok


def check_imports() -> bool:
    ok = True
    for module_name in MODULES_TO_CHECK:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            ok = False
            print(f"[error] import {module_name}: {exc}")
        else:
            print(f"[ok] import {module_name}")
    return ok


def check_streamlit_secrets() -> bool:
    try:
        sections = list(st.secrets.keys())
    except Exception as exc:
        print(f"[error] Streamlit secrets parse failed: {exc}")
        return False

    print(f"[ok] Streamlit secrets sections: {', '.join(sections) or 'none'}")
    return True


def check_app_config() -> None:
    config = load_config()
    print(f"[ok] chat provider: {config.provider}")
    print(f"[ok] embedding provider: {config.embedding_provider}")
    print(f"[ok] OpenAI key present: {bool(config.openai_api_key)}")
    print(f"[ok] OpenAI model: {config.openai_model}")
    print(f"[ok] local base URL: {config.local_base_url}")


def check_chromadb() -> bool:
    config = load_config()
    try:
        embeddings = EmbeddingModel(
            provider=config.embedding_provider,
            openai_api_key=config.openai_api_key,
            openai_embedding_model=config.openai_embedding_model,
            local_base_url=config.local_base_url,
            local_embedding_model=config.local_embedding_model,
        )
        vector_store = VectorStore(config.chroma_path, embeddings)
        documents = vector_store.list_documents()
    except Exception as exc:
        print(f"[error] ChromaDB check failed: {exc}")
        return False

    print(f"[ok] ChromaDB path: {config.chroma_path}")
    print(f"[ok] indexed sources visible: {len(documents)}")
    if vector_store.embedding_warning:
        print(f"[warning] {vector_store.embedding_warning}")
    return True


def main() -> int:
    ok = check_python_runtime()
    ok = check_imports() and ok
    ok = check_streamlit_secrets() and ok
    check_app_config()
    ok = check_chromadb() and ok

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
