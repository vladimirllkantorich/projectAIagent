from __future__ import annotations

import sys
from typing import Tuple


TARGET_PYTHON = "3.11"
SUPPORTED_PYTHON_MIN = (3, 10)
SUPPORTED_PYTHON_MAX_EXCLUSIVE = (3, 14)


def current_python_version() -> str:
    version_info = sys.version_info
    return f"{version_info.major}.{version_info.minor}.{version_info.micro}"


def python_runtime_status() -> Tuple[str, str]:
    current_version = current_python_version()

    if not (SUPPORTED_PYTHON_MIN <= sys.version_info[:2] < SUPPORTED_PYTHON_MAX_EXCLUSIVE):
        return (
            "error",
            (
                "CrewAI requires Python 3.10-3.13. This interpreter is "
                f"Python {current_version}. Recreate `.venv` with `py -3.11 -m venv .venv`."
            ),
        )

    if sys.version_info[:2] != (3, 11):
        return (
            "warning",
            (
                f"Recommended project runtime is Python {TARGET_PYTHON}; current interpreter is "
                f"Python {current_version}. CrewAI supports Python 3.10-3.13."
            ),
        )

    return "ok", f"Python runtime is compatible with CrewAI: {current_version}."
