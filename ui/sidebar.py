from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import streamlit as st

from core.config import AppConfig
from core.index_history import load_index_history


PROVIDERS = ["local", "openai"]
PROVIDER_LABELS = {
    "local": "Local LM Studio",
    "openai": "OpenAI / ChatGPT API",
}
AGENT_MODES = ["standard", "debate"]
AGENT_MODE_LABELS = {
    "standard": "CrewAI standard",
    "debate": "CrewAI debate",
}


@dataclass(frozen=True)
class SidebarState:
    provider: str
    agent_mode: str
    uploaded_files: list
    code_folder_path: str
    index_uploaded: bool
    index_code: bool
    clear_database: bool
    repair_database: bool


def _choose_folder_with_dialog(initial_dir: Optional[str]) -> Optional[str]:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        st.error(f"Folder picker is not available: {exc}")
        return None

    start_dir = initial_dir if initial_dir and Path(initial_dir).is_dir() else str(Path.home())

    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        folder_path = filedialog.askdirectory(
            parent=root,
            title="Choose project folder",
            initialdir=start_dir,
        )
        root.destroy()
    except Exception as exc:
        st.error(f"Could not open the folder picker: {exc}")
        return None

    return folder_path or None


def _select_project_folder() -> str:
    if "project_folder_path" not in st.session_state:
        st.session_state.project_folder_path = ""

    if st.button("Choose project folder", use_container_width=True):
        selected_folder = _choose_folder_with_dialog(st.session_state.project_folder_path)
        if selected_folder:
            st.session_state.project_folder_path = selected_folder

    st.text_input(
        "Project folder path",
        key="project_folder_path",
        placeholder="C:\\path\\to\\project",
    )

    return st.session_state.project_folder_path.strip()


def _select_provider(config: AppConfig) -> str:
    provider = st.selectbox(
        "Model source",
        PROVIDERS,
        index=PROVIDERS.index(config.provider),
        key="model_provider",
        format_func=lambda value: PROVIDER_LABELS[value],
    )
    st.caption(f"Chat provider: {PROVIDER_LABELS[provider]}")

    if provider == "openai" and not config.openai_api_key:
        st.error("Missing `[openai].api_key` in `.streamlit/secrets.toml`.")

    return provider


def _select_agent_mode() -> str:
    return st.selectbox(
        "Agent mode",
        AGENT_MODES,
        index=0,
        key="agent_mode",
        format_func=lambda value: AGENT_MODE_LABELS[value],
    )


def render_sidebar(config: AppConfig) -> SidebarState:
    with st.sidebar:
        st.header("Settings")
        provider = _select_provider(config)
        agent_mode = _select_agent_mode()

        st.header("Indexing")
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=["pdf", "txt", "docx"],
            accept_multiple_files=True,
        )
        index_uploaded = st.button("Index uploaded files", use_container_width=True)

        code_folder_path = _select_project_folder()
        index_code = st.button("Index code folder", use_container_width=True)

        clear_database = st.button("Clear database", use_container_width=True)
        repair_database = st.button("Repair database", use_container_width=True)

    return SidebarState(
        provider=provider,
        agent_mode=agent_mode,
        uploaded_files=uploaded_files or [],
        code_folder_path=code_folder_path,
        index_uploaded=index_uploaded,
        index_code=index_code,
        clear_database=clear_database,
        repair_database=repair_database,
    )


def render_sidebar_documents(document_rows: list[dict]) -> None:
    with st.sidebar:
        _render_document_list(document_rows)
        _render_index_history()


def _render_document_list(document_rows: list[dict]) -> None:
    st.header("Vector database")
    if not document_rows:
        st.caption("No indexed documents yet.")
        return

    for row in document_rows:
        st.caption(f"{row['source']} - {row['chunks']} chunks")


def _render_index_history() -> None:
    st.header("Index history")
    history = load_index_history()
    if not history:
        st.caption("No indexing history yet.")
        return

    for record in reversed(history[-8:]):
        source = record.get("source", "unknown")
        chunks = record.get("chunks", 0)
        status = record.get("status", "indexed")
        source_type = record.get("source_type", "source")
        st.caption(f"{record.get('time', '')} - {status} - {source_type}: {source} ({chunks} chunks)")
