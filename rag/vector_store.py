from __future__ import annotations

from pathlib import Path
import shutil
from typing import Optional
from uuid import uuid4

import chromadb
from chromadb.config import Settings

from core.config import PROJECT_ROOT


DEVVAULT_COLLECTION_PREFIX = "devvault_ai_chunks_"
DATABASE_REPAIR_MARKERS = (
    "mismatched types",
    "metadata segment",
    "backfill request",
    "no such column",
)


class VectorStore:
    def __init__(self, chroma_path: str, embeddings):
        self.chroma_path = Path(chroma_path)
        self.chroma_path.mkdir(parents=True, exist_ok=True)
        self.embeddings = embeddings
        self.collection_name = embeddings.collection_name
        self._warning_message: Optional[str] = None

        try:
            self._open_collection()
        except Exception as exc:
            if not _looks_like_database_repair_error(exc):
                raise RuntimeError(_friendly_chroma_error(exc)) from exc

            warning = _friendly_chroma_error(exc)
            self.repair_database()
            self._warning_message = (
                f"{warning} DevVault AI created a fresh vector database. Re-index your files."
            )

    def _open_collection(self) -> None:
        self.client = _create_persistent_client(self.chroma_path)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def add_chunks(
        self,
        chunks: list[str],
        source: str,
        document_id: str,
        kind: str,
    ) -> None:
        if not chunks:
            return

        ids = [f"{document_id}:{uuid4().hex}" for _ in chunks]
        metadatas = [
            {
                "source": str(source),
                "document_id": str(document_id),
                "kind": kind,
                "chunk_index": index,
            }
            for index, _ in enumerate(chunks)
        ]
        embeddings = self.embeddings.embed_texts(chunks)

        self.delete_document(document_id)

        try:
            self.collection.add(
                ids=ids,
                documents=chunks,
                metadatas=metadatas,
                embeddings=embeddings,
            )
        except Exception as exc:
            raise RuntimeError(_friendly_chroma_error(exc)) from exc

    def search_chunks(
        self,
        query: str,
        n_results: int = 5,
        source: Optional[str] = None,
    ) -> list[dict]:
        try:
            chunk_count = self.collection.count()
        except Exception as exc:
            self._warning_message = _friendly_chroma_error(exc)
            raise RuntimeError(self._warning_message) from exc

        if chunk_count == 0:
            return []

        query_embedding = self.embeddings.embed_texts([query])[0]
        try:
            query_args = {
                "query_embeddings": [query_embedding],
                "n_results": min(n_results, chunk_count),
                "include": ["documents", "metadatas", "distances"],
            }
            if source:
                query_args["where"] = {"source": source}

            result = self.collection.query(**query_args)
        except Exception as exc:
            raise RuntimeError(_friendly_chroma_error(exc)) from exc

        self._warning_message = None
        return _search_rows(result)

    def delete_document(self, document_id: str) -> None:
        try:
            if self.collection.count() == 0:
                return
            self.collection.delete(where={"document_id": str(document_id)})
        except Exception as exc:
            raise RuntimeError(_friendly_chroma_error(exc)) from exc

    def clear_database(self) -> None:
        try:
            for collection_name in _devvault_collection_names(self.client):
                _delete_collection(self.client, collection_name)
            self.collection = self.client.get_or_create_collection(name=self.collection_name)
        except Exception:
            self.repair_database()

        self._warning_message = None

    def repair_database(self) -> None:
        _ensure_safe_repair_path(self.chroma_path)
        _stop_chroma_client(getattr(self, "client", None))
        _clear_chroma_system_cache()
        if self.chroma_path.exists():
            shutil.rmtree(self.chroma_path)
        self.chroma_path.mkdir(parents=True, exist_ok=True)
        self._open_collection()
        self._warning_message = None

    def list_documents(self) -> list[dict]:
        try:
            if self.collection.count() == 0:
                return []

            items = self.collection.get(include=["metadatas"])
        except Exception as exc:
            self._warning_message = _friendly_chroma_error(exc)
            return []

        return _document_rows(items.get("metadatas", []))

    @property
    def embedding_warning(self) -> Optional[str]:
        warnings = [
            warning
            for warning in (self._warning_message, self.embeddings.warning_message)
            if warning
        ]
        return "\n\n".join(warnings) if warnings else None


def _create_persistent_client(chroma_path: Path):
    settings = Settings(anonymized_telemetry=False)
    try:
        return chromadb.PersistentClient(path=str(chroma_path), settings=settings)
    except ValueError as exc:
        if "different settings" not in str(exc):
            raise

        # Streamlit hot reload can leave an older Chroma singleton alive for the
        # same path after code changes. Clear it and retry with the current app settings.
        _clear_chroma_system_cache()
        return chromadb.PersistentClient(path=str(chroma_path), settings=settings)


def _clear_chroma_system_cache() -> None:
    try:
        from chromadb.api.shared_system_client import SharedSystemClient
    except ModuleNotFoundError:
        from chromadb.api.client import SharedSystemClient

    SharedSystemClient.clear_system_cache()


def _stop_chroma_client(client) -> None:
    if client is None:
        return

    try:
        client._system.stop()
    except Exception:
        pass


def _delete_collection(client, collection_name: str) -> None:
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass


def _devvault_collection_names(client) -> list[str]:
    collection_names = []
    for collection in client.list_collections():
        name = collection if isinstance(collection, str) else getattr(collection, "name", "")
        if name.startswith(DEVVAULT_COLLECTION_PREFIX):
            collection_names.append(name)
    return collection_names


def _search_rows(result) -> list[dict]:
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    return [
        {
            "text": text,
            "metadata": metadata or {},
            "distance": distance,
            "score": _score_from_distance(distance),
        }
        for text, metadata, distance in zip(documents, metadatas, distances)
    ]


def _document_rows(metadatas: list[dict]) -> list[dict]:
    document_counts: dict[str, int] = {}

    for metadata in metadatas:
        source = (metadata or {}).get("source", "unknown")
        document_counts[source] = document_counts.get(source, 0) + 1

    return [
        {"source": source, "chunks": chunk_count}
        for source, chunk_count in sorted(document_counts.items())
    ]


def _score_from_distance(distance) -> Optional[float]:
    try:
        numeric_distance = float(distance)
    except (TypeError, ValueError):
        return None

    return 1 / (1 + max(numeric_distance, 0.0))


def _ensure_safe_repair_path(chroma_path: Path) -> None:
    project_root = PROJECT_ROOT.resolve()
    target = chroma_path.resolve()
    if target == project_root or project_root not in target.parents:
        raise RuntimeError(f"Refusing to repair ChromaDB path outside the project: {target}")


def _friendly_chroma_error(exc: Exception) -> str:
    message = str(exc)
    lowered = message.lower()

    if "dimension" in lowered:
        return (
            "ChromaDB rejected the embeddings because this collection was created with a different "
            "embedding size. This usually happens after changing the embedding model or switching "
            "between API embeddings and fallback embeddings. Clear the database and index the files again."
        )

    if _contains_any(lowered, DATABASE_REPAIR_MARKERS):
        return (
            "ChromaDB could not read the current vector database. It may be corrupted or from an "
            "incompatible ChromaDB version. Click `Clear database`, then index the files again."
        )

    return f"ChromaDB operation failed: {message}"


def _looks_like_database_repair_error(exc: Exception) -> bool:
    return _contains_any(str(exc).lower(), DATABASE_REPAIR_MARKERS)


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)
