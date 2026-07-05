from __future__ import annotations

import hashlib
from typing import Optional

import streamlit as st

from agents.supervisor import SupervisorAgent
from core.config import AppConfig, load_config
from core.index_history import clear_index_history
from core.llm_client import LLMClient
from core.python_runtime import python_runtime_status
from rag.embeddings import EmbeddingModel
from rag.vector_store import VectorStore
from tools.indexing_tools import index_code_folder, index_uploaded_files
from ui.agent_log import clear_agent_log, init_agent_log, log_agent_event, render_agent_log
from ui.chat import add_message, clear_chat_history, conversation_memory, init_chat_state, render_chat_messages
from ui.sidebar import render_sidebar, render_sidebar_documents


st.set_page_config(page_title="DevVault AI", page_icon="DV", layout="wide")

CHROMA_CLIENT_CACHE_VERSION = "shared-rag-embedding-provider-v3"

SECRETS_HELP = """
Create `.streamlit/secrets.toml`. DevVault AI reads the real secrets file, not `secrets.toml.example`.

`.streamlit/secrets.toml` example:

```toml
[openai]
api_key = ""
model = "gpt-5-mini"

[local]
base_url = "http://localhost:1234/v1"
model = "local-model"

[app]
default_provider = "local"
embedding_provider = "local"
chroma_path = "./data/chroma_db"
max_parallel_agents = 1
```
""".strip()


@st.cache_resource(show_spinner=False)
def get_vector_store(
    chroma_path: str,
    embedding_provider: str,
    _openai_api_key: Optional[str],
    openai_api_key_fingerprint: str,
    openai_embedding_model: str,
    local_base_url: str,
    local_embedding_model: str,
    cache_version: str,
) -> VectorStore:
    # Fingerprint and cache_version are intentionally part of Streamlit's cache key.
    embeddings = EmbeddingModel(
        provider=embedding_provider,
        openai_api_key=_openai_api_key,
        openai_embedding_model=openai_embedding_model,
        local_base_url=local_base_url,
        local_embedding_model=local_embedding_model,
    )
    return VectorStore(chroma_path, embeddings)


def openai_key_fingerprint(openai_api_key: Optional[str]) -> str:
    if not openai_api_key:
        return "missing"
    return hashlib.sha256(openai_api_key.encode("utf-8")).hexdigest()[:12]


def reset_database(vector_store: VectorStore, repair: bool = False) -> None:
    clear_agent_log()
    if repair:
        vector_store.repair_database()
        toast_message = "Vector database repaired."
    else:
        vector_store.clear_database()
        toast_message = "Vector database cleared."

    clear_index_history()
    st.toast(toast_message)
    st.rerun()


def reset_chat() -> None:
    clear_chat_history()
    clear_agent_log()
    st.toast("Chat cleared.")
    st.rerun()


def show_sidebar_embedding_warning(vector_store: VectorStore) -> None:
    if vector_store.embedding_warning:
        st.sidebar.warning(vector_store.embedding_warning)


def run_indexing_job(
    vector_store: VectorStore,
    spinner_text: str,
    success_text: str,
    error_prefix: str,
    job,
) -> None:
    try:
        with st.spinner(spinner_text):
            chunk_count = job()
    except Exception as exc:
        st.sidebar.error(f"{error_prefix}: {exc}")
        return

    show_sidebar_embedding_warning(vector_store)
    st.sidebar.success(success_text.format(chunk_count=chunk_count))
    st.rerun()


def handle_sidebar_actions(sidebar_state, vector_store: VectorStore) -> None:
    if sidebar_state.clear_database:
        reset_database(vector_store)
        return

    if sidebar_state.repair_database:
        reset_database(vector_store, repair=True)
        return

    if sidebar_state.index_uploaded:
        if not sidebar_state.uploaded_files:
            st.sidebar.warning("Upload PDF, TXT, or DOCX files before indexing.")
            return

        run_indexing_job(
            vector_store=vector_store,
            spinner_text="Indexing uploaded files...",
            success_text="Indexed {chunk_count} chunks from uploaded files.",
            error_prefix="Document indexing failed",
            job=lambda: index_uploaded_files(sidebar_state.uploaded_files, vector_store),
        )
        return

    if sidebar_state.index_code:
        if not sidebar_state.code_folder_path:
            st.sidebar.warning("Choose a project folder before indexing.")
            return

        run_indexing_job(
            vector_store=vector_store,
            spinner_text="Indexing code folder...",
            success_text="Indexed {chunk_count} chunks from the code folder.",
            error_prefix="Code indexing failed",
            job=lambda: index_code_folder(sidebar_state.code_folder_path, vector_store),
        )


def answer_user_question(
    question: str,
    config: AppConfig,
    vector_store: VectorStore,
    agent_mode: str,
    memory: str,
) -> str:
    active_config = config
    llm_client = LLMClient(active_config)

    try:
        llm_client.validate_provider()
    except Exception as exc:
        fallback_client = _openai_fallback_client(active_config, exc)
        if fallback_client:
            active_config, llm_client = fallback_client
            log_agent_event(
                "Supervisor Agent",
                "Local LM Studio is not ready, so I switched this answer to OpenAI / ChatGPT API.",
                status="warning",
            )
        else:
            log_agent_event(
                "Supervisor Agent",
                "Provider validation failed before starting the agent workflow.",
                status="error",
            )
            return str(exc)

    provider_name = "OpenAI / ChatGPT API" if active_config.provider == "openai" else "Local LM Studio"
    model_name = llm_client.chat_model_name()
    log_agent_event(
        "Supervisor Agent",
        f"Using {provider_name} as the chat provider. Model: {model_name}. Agent mode: {agent_mode}.",
        status="info",
    )

    supervisor = SupervisorAgent(
        llm_client=llm_client,
        vector_store=vector_store,
        agent_log=st.session_state.agent_log,
        agent_mode=agent_mode,
        conversation_memory=memory,
    )
    return supervisor.answer(question)


def _openai_fallback_client(config: AppConfig, provider_error: Exception):
    error_text = str(provider_error).lower()
    if config.provider != "local" or "lm studio" not in error_text:
        return None
    if not config.openai_api_key:
        return None

    fallback_config = config.with_overrides(provider="openai")
    fallback_client = LLMClient(fallback_config)
    try:
        fallback_client.validate_provider()
    except Exception:
        return None

    return fallback_config, fallback_client


def stop_if_python_version_is_wrong() -> None:
    runtime_status, runtime_message = python_runtime_status()
    if runtime_status == "error":
        st.error(runtime_message)
        st.stop()
    if runtime_status == "warning":
        st.warning(runtime_message)


def create_vector_store(config: AppConfig) -> VectorStore:
    return get_vector_store(
        config.chroma_path,
        config.embedding_provider,
        config.openai_api_key,
        openai_key_fingerprint(config.openai_api_key),
        config.openai_embedding_model,
        config.local_base_url,
        config.local_embedding_model,
        CHROMA_CLIENT_CACHE_VERSION,
    )


def show_setup_messages(base_config: AppConfig, active_config: AppConfig) -> None:
    if not base_config.secrets_found:
        st.info(
            f"No Streamlit secrets file was found at `{base_config.secrets_path}`. "
            "Local mode can still run with defaults, but create this file before using OpenAI mode."
        )
        with st.expander("secrets.toml example"):
            st.markdown(SECRETS_HELP)

    if active_config.provider == "openai" and not active_config.openai_api_key:
        st.error(
            "OpenAI API key is missing. Create `.streamlit/secrets.toml` and add "
            "`[openai] api_key = \"sk-...\"`, or choose Local LM Studio in the sidebar."
        )


def handle_chat_question(
    active_config: AppConfig,
    vector_store: VectorStore,
    agent_mode: str,
) -> None:
    question = st.chat_input("Ask about your documents, code, or project plan")
    if not question:
        return

    memory = conversation_memory()
    add_message("user", question)
    clear_agent_log()

    with st.spinner("Agents are working..."):
        answer = answer_user_question(
            question,
            active_config,
            vector_store,
            agent_mode,
            memory,
        )

    add_message("assistant", answer)


def render_chat_and_log() -> None:
    chat_col, log_col = st.columns([0.62, 0.38], gap="large")

    with chat_col:
        if render_chat_messages():
            reset_chat()

    with log_col:
        render_agent_log()


def main() -> None:
    init_chat_state()
    init_agent_log()

    st.title("DevVault AI")
    stop_if_python_version_is_wrong()

    base_config = load_config()
    sidebar_state = render_sidebar(base_config)
    active_config = base_config.with_overrides(provider=sidebar_state.provider)
    vector_store = create_vector_store(active_config)
    document_rows = vector_store.list_documents()

    handle_sidebar_actions(sidebar_state, vector_store)
    render_sidebar_documents(document_rows)

    if vector_store.embedding_warning:
        st.warning(vector_store.embedding_warning)

    show_setup_messages(base_config, active_config)
    handle_chat_question(active_config, vector_store, sidebar_state.agent_mode)
    render_chat_and_log()


if __name__ == "__main__":
    main()
