from __future__ import annotations

import json
import re

from core.prompts import DOCUMENT_AGENT_PROMPT
from tools.search_tools import format_search_context


class DocumentAgent:
    def __init__(self, llm_client, logger=None):
        self.llm_client = llm_client
        self.logger = logger

    def answer(
        self,
        question: str,
        search_results: list[dict],
        conversation_memory: str = "",
    ) -> str:
        return self.answer_structured(question, search_results, conversation_memory)["answer"]

    def answer_structured(
        self,
        question: str,
        search_results: list[dict],
        conversation_memory: str = "",
    ) -> dict:
        if self.logger:
            self.logger("Document Agent", "Reading the most relevant document chunks.")

        if not search_results:
            return _empty_document_result("I could not find that information in the uploaded documents or indexed code.")

        context = format_search_context(search_results)
        messages = [
            {"role": "system", "content": DOCUMENT_AGENT_PROMPT},
            {
                "role": "user",
                "content": (
                    f"{_memory_section(conversation_memory)}"
                    f"Question:\n{question}\n\n"
                    f"Relevant document chunks:\n{context or 'No matching document chunks were found.'}"
                ),
            },
        ]
        response_text = self.llm_client.chat(messages)
        parsed = _parse_json_object(response_text)
        return _normalize_document_result(parsed, search_results, response_text)


def _memory_section(conversation_memory: str) -> str:
    if not conversation_memory:
        return ""

    return (
        "Conversation memory for follow-up context only, not as document evidence:\n"
        f"{conversation_memory}\n\n"
    )


def _parse_json_object(text: str) -> dict:
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}

    try:
        parsed = json.loads(match.group(0))
    except Exception:
        return {}

    return parsed if isinstance(parsed, dict) else {}


def _normalize_document_result(parsed: dict, search_results: list[dict], raw_answer: str) -> dict:
    parsed_was_valid = bool(parsed)
    coverage = str(parsed.get("coverage", "")).strip().lower()
    if coverage not in {"none", "partial", "full"}:
        coverage = "partial" if search_results else "none"

    evidence = parsed.get("evidence", [])
    if not isinstance(evidence, list):
        evidence = []
    evidence = [_normalize_evidence(item, search_results) for item in evidence if isinstance(item, dict)]
    evidence = [item for item in evidence if item]

    sources = parsed.get("sources", [])
    if not isinstance(sources, list):
        sources = []
    source_names = _source_names(search_results)
    sources = sorted({str(source) for source in sources if str(source) in source_names})

    found_in_context = _as_bool(parsed.get("found_in_context", False))
    if not parsed_was_valid and search_results and raw_answer and not _looks_like_not_found(raw_answer):
        found_in_context = True
        coverage = "partial"
    if evidence and coverage != "none":
        found_in_context = True
    if found_in_context and coverage != "none" and not evidence:
        evidence = [_fallback_evidence(search_results)]
    if not found_in_context:
        coverage = "none"

    answer = str(parsed.get("answer") or raw_answer or "").strip()
    if not answer:
        answer = "The retrieved documents do not contain information about this concept."

    return {
        "found_in_context": found_in_context,
        "coverage": coverage,
        "answer": answer,
        "evidence": evidence,
        "sources": sources or (source_names if found_in_context else []),
    }


def _normalize_evidence(item: dict, search_results: list[dict]) -> dict:
    source_names = _source_names(search_results)
    source = str(item.get("source", "")).strip()
    if source not in source_names:
        source = source_names[0] if source_names else "unknown"

    quote_or_summary = str(item.get("quote_or_summary", "")).strip()
    if not quote_or_summary:
        quote_or_summary = _first_chunk_summary(search_results)

    chunk_id = str(item.get("chunk_id", "")).strip()
    if not chunk_id:
        chunk_id = _first_chunk_id(search_results)

    return {
        "source": source,
        "chunk_id": chunk_id,
        "quote_or_summary": quote_or_summary,
    }


def _fallback_evidence(search_results: list[dict]) -> dict:
    source_names = _source_names(search_results)
    return {
        "source": source_names[0] if source_names else "unknown",
        "chunk_id": _first_chunk_id(search_results),
        "quote_or_summary": _first_chunk_summary(search_results),
    }


def _empty_document_result(answer: str) -> dict:
    return {
        "found_in_context": False,
        "coverage": "none",
        "answer": answer,
        "evidence": [],
        "sources": [],
    }


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return bool(value)


def _looks_like_not_found(text: str) -> bool:
    lowered = text.lower()
    markers = (
        "does not contain",
        "do not contain",
        "no information",
        "not found",
        "could not find",
        "missing from the context",
    )
    return any(marker in lowered for marker in markers)


def _source_names(search_results: list[dict]) -> list[str]:
    return sorted(
        {
            str((item.get("metadata") or {}).get("source", "unknown"))
            for item in search_results
        }
    )


def _first_chunk_id(search_results: list[dict]) -> str:
    if not search_results:
        return ""

    metadata = search_results[0].get("metadata") or {}
    document_id = metadata.get("document_id", "chunk")
    chunk_index = metadata.get("chunk_index", 0)
    return f"{document_id}:{chunk_index}"


def _first_chunk_summary(search_results: list[dict]) -> str:
    if not search_results:
        return ""

    text = str(search_results[0].get("text", "")).strip()
    return text[:240]
