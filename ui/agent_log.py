from __future__ import annotations

from datetime import datetime
from typing import Optional

import streamlit as st


KNOWN_STATUSES = {"info", "running", "done", "warning", "error"}


def init_agent_log() -> None:
    if "agent_log" not in st.session_state:
        st.session_state.agent_log = []


def clear_agent_log() -> None:
    st.session_state.agent_log = []


def log_agent_event(
    agent_name: str,
    message: str,
    status: str = "info",
    elapsed_seconds: Optional[float] = None,
) -> None:
    init_agent_log()
    st.session_state.agent_log.append(
        {
            "agent": agent_name,
            "message": message,
            "status": status if status in KNOWN_STATUSES else "info",
            "elapsed_seconds": elapsed_seconds,
            "time": datetime.now().strftime("%H:%M:%S"),
        }
    )


def render_agent_log() -> None:
    st.subheader("Agent Chat / Agent Log")

    if not st.session_state.agent_log:
        st.caption("No agent activity yet.")
        return

    for item in st.session_state.agent_log:
        status = item.get("status", "info")
        elapsed_seconds = item.get("elapsed_seconds")
        elapsed_text = f" - {elapsed_seconds:.2f}s" if elapsed_seconds is not None else ""
        st.markdown(f"**{item.get('agent', 'Agent')}** `{status}` `{item.get('time', '')}{elapsed_text}`")
        st.write(item["message"])
