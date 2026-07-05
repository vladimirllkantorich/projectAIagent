from __future__ import annotations

from tools.search_tools import search_relevant_chunks


class SearchAgent:
    def __init__(self, vector_store, logger=None):
        self.vector_store = vector_store
        self.logger = logger

    def search(self, question: str, n_results: int = 8) -> list[dict]:
        search_result = search_relevant_chunks(
            self.vector_store,
            question,
            n_results=n_results,
        )

        if self.logger:
            _log_search_results(
                self.logger,
                search_result.chunks,
                search_result.weak_match_count,
                search_result.source_filter,
            )

        return search_result.chunks


def _log_search_results(
    logger,
    results: list[dict],
    weak_match_count: int,
    source_filter: str = "",
) -> None:
    if not results:
        if source_filter:
            logger(
                "Search Agent",
                f"Searched only in `{source_filter}`, but no matching chunks were found.",
                status="warning",
            )
            return

        logger(
            "Search Agent",
            "No strong matching chunks were found in the vector database.",
            status="warning",
        )
        return

    sources = sorted(
        {
            (item.get("metadata") or {}).get("source", "unknown")
            for item in results
        }
    )
    logger(
        "Search Agent",
        _search_log_message(results, sources, weak_match_count, source_filter),
        status="done",
    )


def _search_log_message(
    results: list[dict],
    sources: list[str],
    weak_match_count: int,
    source_filter: str,
) -> str:
    if source_filter:
        return f"Searched only in `{source_filter}` and found {len(results)} chunk(s)."

    return (
        f"Found {len(results)} relevant chunks from {len(sources)} source(s): {', '.join(sources)}. "
        f"Filtered out {weak_match_count} weak matches."
    )
