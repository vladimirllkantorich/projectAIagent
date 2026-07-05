from __future__ import annotations

import streamlit as st


WELCOME_MESSAGE = "Hi, I am DevVault AI. Upload or index files, then ask me about them."


def _initial_messages() -> list[dict[str, str]]:
    return [{"role": "assistant", "content": WELCOME_MESSAGE}]


def init_chat_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = _initial_messages()


def clear_chat_history() -> None:
    st.session_state.messages = _initial_messages()


def chat_has_history() -> bool:
    return len(st.session_state.get("messages", [])) > 1


def add_message(role: str, content: str) -> None:
    st.session_state.messages.append({"role": role, "content": content})


def conversation_memory(limit: int = 6) -> str:
    messages = st.session_state.get("messages", [])
    recent_messages = messages[-limit:]
    memory_lines = []

    for message in recent_messages:
        role = message.get("role", "unknown")
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        memory_lines.append(f"{role}: {content}")

    return "\n".join(memory_lines)


def render_chat_messages() -> bool:
    title_col, action_col = st.columns([0.72, 0.28])
    with title_col:
        st.subheader("Chat")
    with action_col:
        clear_clicked = st.button(
            "Clear chat",
            use_container_width=True,
            disabled=not chat_has_history(),
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    return clear_clicked
