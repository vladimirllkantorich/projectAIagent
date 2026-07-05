from __future__ import annotations

from typing import Optional

from core.prompts import SUMMARY_AGENT_PROMPT


MAX_ANSWER_LENGTH_WITHOUT_SUMMARY = 3500


class SummaryAgent:
    def __init__(self, llm_client, logger=None):
        self.llm_client = llm_client
        self.logger = logger

    def shorten_if_needed(self, question: str, answer: str) -> str:
        if len(answer) < MAX_ANSWER_LENGTH_WITHOUT_SUMMARY:
            if self.logger:
                self.logger("Summary Agent", "The answer is already concise enough.")
            return self.guard_contradictions(answer)

        if self.logger:
            self.logger("Summary Agent", "Shortening the long draft answer.")

        messages = [
            {"role": "system", "content": SUMMARY_AGENT_PROMPT},
            {
                "role": "user",
                "content": f"Question:\n{question}\n\nLong answer:\n{answer}",
            },
        ]
        return self.guard_contradictions(self.llm_client.chat(messages))

    def synthesize_document_answer(
        self,
        question: str,
        document_result: dict,
        search_results: list[dict],
    ) -> str:
        if self.logger:
            self.logger("Summary Agent", "Resolving document coverage before writing the final answer.")

        coverage = str(document_result.get("coverage", "none")).lower()
        found_in_context = _as_bool(document_result.get("found_in_context", False))
        evidence_text = _format_evidence(document_result)
        document_answer = str(document_result.get("answer", "")).strip()

        if found_in_context and coverage == "full":
            status = "The retrieved documents explain this concept."
        elif found_in_context:
            status = "The retrieved documents contain a partial mention of the concept, but not a full definition."
        else:
            status = "The retrieved documents do not contain information about this concept."

        messages = [
            {"role": "system", "content": SUMMARY_AGENT_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    f"Structured document status:\n"
                    f"- found_in_context: {found_in_context}\n"
                    f"- coverage: {coverage}\n"
                    f"- required opening status: {status}\n\n"
                    f"Document Agent answer:\n{document_answer}\n\n"
                    f"Evidence:\n{evidence_text or 'No source evidence.'}\n\n"
                    "Write one coherent final answer. Start with the required opening status. "
                    "Do not say the documents contain no information if found_in_context is true."
                ),
            },
        ]

        try:
            answer = self.llm_client.chat(messages)
        except Exception:
            answer = _deterministic_document_answer(status, document_answer, evidence_text, found_in_context)

        return self.guard_contradictions(answer, document_result=document_result, search_results=search_results)

    def guard_contradictions(
        self,
        answer: str,
        document_result: Optional[dict] = None,
        search_results: Optional[list[dict]] = None,
    ) -> str:
        answer_text = (answer or "").strip()
        if not _has_not_found_and_document_evidence(answer_text):
            return answer_text

        if document_result and _as_bool(document_result.get("found_in_context")):
            coverage = str(document_result.get("coverage", "partial")).lower()
            status = (
                "The retrieved documents explain this concept."
                if coverage == "full"
                else "The retrieved documents contain a partial mention of the concept, but not a full definition."
            )
            evidence_text = _format_evidence(document_result)
            document_answer = str(document_result.get("answer", "")).strip()
            return _deterministic_document_answer(status, document_answer, evidence_text, True)

        if search_results:
            return _partial_context_answer(search_results)

        return "The retrieved documents do not contain information about this concept."


def _has_not_found_and_document_evidence(answer: str) -> bool:
    lowered = answer.lower()
    partial_ok = "partial mention" in lowered and "not a full definition" in lowered
    if partial_ok:
        return False

    not_found_markers = (
        "does not contain information",
        "do not contain information",
        "no information",
        "missing from the context",
        "not found in the context",
        "not contain any information",
    )
    evidence_markers = (
        "based on the document",
        "the document provides",
        "according to the document",
        "examples include",
        "retrieved documents explain",
        "retrieved chunks",
    )
    return any(marker in lowered for marker in not_found_markers) and any(
        marker in lowered for marker in evidence_markers
    )


def _format_evidence(document_result: dict) -> str:
    evidence = document_result.get("evidence") or []
    lines = []
    for item in evidence[:4]:
        if not isinstance(item, dict):
            continue
        source = item.get("source", "unknown")
        chunk_id = item.get("chunk_id", "")
        quote_or_summary = item.get("quote_or_summary", "")
        chunk_text = f" ({chunk_id})" if chunk_id else ""
        lines.append(f"- {source}{chunk_text}: {quote_or_summary}")
    return "\n".join(lines)


def _deterministic_document_answer(
    status: str,
    document_answer: str,
    evidence_text: str,
    found_in_context: bool,
) -> str:
    parts = [status]

    if document_answer:
        parts.append(document_answer)

    if found_in_context and evidence_text:
        parts.append(f"Evidence:\n{evidence_text}")

    return "\n\n".join(parts).strip()


def _partial_context_answer(search_results: list[dict]) -> str:
    sources = sorted(
        {
            str((item.get("metadata") or {}).get("source", "unknown"))
            for item in search_results
        }
    )
    source_text = ", ".join(sources) if sources else "the retrieved sources"
    return (
        "The retrieved documents contain a partial mention of the concept, but not a full definition.\n\n"
        f"Relevant retrieved context was found in: {source_text}."
    )


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return bool(value)
