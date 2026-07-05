from __future__ import annotations

from dataclasses import dataclass, replace
import os
from pathlib import Path
from typing import Any, Optional

import streamlit as st


VALID_PROVIDERS = {"local", "lm_studio", "lmstudio", "openai"}
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SECRETS_PATH = PROJECT_ROOT / ".streamlit" / "secrets.toml"
ENV_PATH = PROJECT_ROOT / ".env"


@dataclass(frozen=True)
class AppConfig:
    provider: str
    embedding_provider: str
    openai_api_key: Optional[str]
    openai_model: str
    openai_embedding_model: str
    local_base_url: str
    local_model: str
    local_embedding_model: str
    chroma_path: str
    max_parallel_agents: int
    secrets_path: str
    secrets_found: bool

    def with_overrides(self, **overrides: Any) -> "AppConfig":
        cleaned = {key: value for key, value in overrides.items() if value is not None}
        if "provider" in cleaned:
            cleaned["provider"] = _as_provider(cleaned["provider"])
        if "embedding_provider" in cleaned:
            cleaned["embedding_provider"] = _as_provider(cleaned["embedding_provider"])
        return replace(self, **cleaned)


def _secrets_section(section_name: str) -> dict[str, Any]:
    try:
        section = st.secrets.get(section_name, {})
    except Exception:
        return {}

    if section is None:
        return {}

    return dict(section)


def _as_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _as_text(value: Any, fallback: str) -> str:
    if value is None:
        return fallback

    text = str(value).strip()
    return text or fallback


def _as_optional_text(value: Any) -> Optional[str]:
    if value is None:
        return None

    text = str(value).strip()
    return text or None


def _as_provider(value: Any) -> str:
    provider = _as_text(value, "local").lower()
    if provider not in VALID_PROVIDERS:
        return "local"
    if provider in {"lm_studio", "lmstudio"}:
        return "local"
    return provider


def _project_path(value: Any, fallback: str) -> str:
    text = _as_text(value, fallback)
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path)


def _env_file_values() -> dict[str, str]:
    if not ENV_PATH.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in ENV_PATH.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _config_value(
    env_values: dict[str, str],
    env_key: str,
    section: dict[str, Any],
    section_key: str,
    fallback: Any = None,
) -> Any:
    return section.get(section_key) or os.getenv(env_key) or env_values.get(env_key) or fallback


def load_config() -> AppConfig:
    env_values = _env_file_values()
    openai_settings = _secrets_section("openai")
    local_settings = _secrets_section("local")
    app_settings = _secrets_section("app")

    return AppConfig(
        provider=_as_provider(
            _config_value(env_values, "APP_DEFAULT_PROVIDER", app_settings, "default_provider")
        ),
        embedding_provider=_as_provider(
            _config_value(env_values, "APP_EMBEDDING_PROVIDER", app_settings, "embedding_provider")
        ),
        openai_api_key=_as_optional_text(
            _config_value(env_values, "OPENAI_API_KEY", openai_settings, "api_key")
        ),
        openai_model=_as_text(
            _config_value(env_values, "OPENAI_MODEL", openai_settings, "model"),
            "gpt-5-mini",
        ),
        openai_embedding_model=_as_text(
            _config_value(env_values, "OPENAI_EMBEDDING_MODEL", openai_settings, "embedding_model"),
            "text-embedding-3-small",
        ),
        local_base_url=_as_text(
            _config_value(env_values, "LM_STUDIO_BASE_URL", local_settings, "base_url"),
            "http://localhost:1234/v1",
        ),
        local_model=_as_text(
            _config_value(env_values, "LM_STUDIO_MODEL", local_settings, "model"),
            "local-model",
        ),
        local_embedding_model=_as_text(
            _config_value(env_values, "LM_STUDIO_EMBEDDING_MODEL", local_settings, "embedding_model"),
            "nomic-embed-text-v1.5",
        ),
        chroma_path=_project_path(
            _config_value(env_values, "APP_CHROMA_PATH", app_settings, "chroma_path"),
            "./data/chroma_db",
        ),
        max_parallel_agents=max(
            1,
            _as_int(
                _config_value(env_values, "APP_MAX_PARALLEL_AGENTS", app_settings, "max_parallel_agents", 1),
                1,
            ),
        ),
        secrets_path=str(SECRETS_PATH),
        secrets_found=SECRETS_PATH.exists() or ENV_PATH.exists(),
    )
