from __future__ import annotations

from pathlib import Path


ALLOWED_EXTENSIONS = {
    ".bat",
    ".css",
    ".go",
    ".html",
    ".java",
    ".js",
    ".jsx",
    ".json",
    ".md",
    ".ps1",
    ".py",
    ".rs",
    ".sh",
    ".sql",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
}
ALLOWED_FILENAMES = {".env.example"}
IGNORED_DIRS = {
    "__pycache__",
    ".git",
    ".idea",
    ".venv",
    "venv",
    "node_modules",
    "chroma_db",
    "uploaded_files",
}
SENSITIVE_FILENAMES = {".env", "secrets.toml"}


def load_code_project(folder_path: str, max_file_size: int = 500_000) -> list[dict[str, str]]:
    root = Path(folder_path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Code folder does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Code folder path is not a directory: {root}")

    files = []
    for file_path in sorted(root.rglob("*")):
        relative_path = file_path.relative_to(root)
        if not _should_read_file(file_path, relative_path, max_file_size):
            continue

        text = file_path.read_text(encoding="utf-8", errors="replace")
        files.append(
            {
                "path": str(file_path),
                "source": str(relative_path),
                "text": text,
            }
        )

    return files


def _should_read_file(file_path: Path, relative_path: Path, max_file_size: int) -> bool:
    if not file_path.is_file():
        return False
    if _is_ignored_path(relative_path):
        return False
    if not _is_allowed_file(file_path):
        return False
    return file_path.stat().st_size <= max_file_size


def _is_ignored_path(relative_path: Path) -> bool:
    parts = {part.lower() for part in relative_path.parts}
    return bool(parts & IGNORED_DIRS) or relative_path.name.lower() in SENSITIVE_FILENAMES


def _is_allowed_file(file_path: Path) -> bool:
    return file_path.name in ALLOWED_FILENAMES or file_path.suffix.lower() in ALLOWED_EXTENSIONS
