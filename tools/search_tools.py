from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePath


MIN_RELEVANCE_SCORE = 0.20


@dataclass(frozen=True)
class SearchResult:
    chunks: list[dict]
    weak_match_count: int
    source_filter: str = ""


def search_relevant_chunks(vector_store, question: str, n_results: int = 8) -> SearchResult:
    source_filter = find_requested_source(question, available_sources(vector_store))
    if source_filter:
        raw_results = vector_store.search_chunks(
            question,
            n_results=n_results,
            source=source_filter,
        )
        return SearchResult(
            chunks=raw_results,
            weak_match_count=0,
            source_filter=source_filter,
        )

    raw_results = vector_store.search_chunks(question, n_results=n_results)
    results = [
        item
        for item in raw_results
        if is_relevant(item)
    ]
    weak_match_count = len(raw_results) - len(results)
    return SearchResult(
        chunks=results,
        weak_match_count=weak_match_count,
    )


def available_sources(vector_store) -> list[str]:
    return [
        row["source"]
        for row in vector_store.list_documents()
        if row.get("source")
    ]


def find_requested_source(question: str, sources: list[str]) -> str:
    if not question or not sources:
        return ""

    normalized_question = _normalize_path_text(question)
    full_path_match = _find_full_path_match(normalized_question, sources)
    if full_path_match:
        return full_path_match

    basename_matches = [
        source
        for source in sources
        if _source_basename(source) in normalized_question
    ]
    return basename_matches[0] if len(basename_matches) == 1 else ""


def is_relevant(result: dict) -> bool:
    score = result.get("score")
    return score is None or score >= MIN_RELEVANCE_SCORE


def format_search_context(search_results: list[dict], source_label: str = "Source") -> str:
    parts = []
    for index, item in enumerate(search_results, start=1):
        metadata = item.get("metadata", {})
        source = metadata.get("source", "unknown")
        score = item.get("score")
        score_text = f" | relevance: {score:.2f}" if score is not None else ""
        text = item.get("text", "")
        parts.append(f"[{index}] {source_label}: {source}{score_text}\n{text}")
    return "\n\n".join(parts)


def format_source_footer(search_results: list[dict]) -> str:
    source_counts = {}

    for item in search_results:
        source = item.get("metadata", {}).get("source", "unknown")
        source_counts[source] = source_counts.get(source, 0) + 1

    if not source_counts:
        return ""

    lines = ["Sources:"]
    for source, count in sorted(source_counts.items()):
        lines.append(f"- {source} ({count} chunks)")

    return "\n".join(lines)


def _find_full_path_match(normalized_question: str, sources: list[str]) -> str:
    matches = [
        source
        for source in sources
        if _normalize_path_text(source) in normalized_question
    ]
    return max(matches, key=len) if matches else ""


def _source_basename(source: str) -> str:
    normalized_source = _normalize_path_text(source)
    return PurePath(normalized_source).name


def _normalize_path_text(text: str) -> str:
    return str(text).replace("\\", "/").casefold()
