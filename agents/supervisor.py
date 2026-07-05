from __future__ import annotations

from datetime import datetime
import time
from typing import Optional

from agents.crewai_pipeline import CrewAIRagPipeline
from agents.document_agent import DocumentAgent
from agents.search_agent import SearchAgent
from agents.summary_agent import SummaryAgent
from agents.task_agent import TaskAgent
from core.llm_client import LLMClientError
from tools.search_tools import format_source_footer


class SupervisorAgent:
    def __init__(
        self,
        llm_client,
        vector_store,
        agent_log: list[dict],
        agent_mode: str = "standard",
        conversation_memory: str = "",
    ):
        self.llm_client = llm_client
        self.vector_store = vector_store
        self.agent_log = agent_log
        self.agent_mode = agent_mode
        self.conversation_memory = conversation_memory

    def answer(self, question: str) -> str:
        self._log("Supervisor Agent", "Reading the request and choosing the CrewAI route.", status="running")

        try:
            plan = self._make_plan(question)
            search_results = self._search(question)
            if _should_use_simple_document_route(question, plan.task_type, search_results, self.agent_mode):
                final_answer = self._answer_with_document_summary(question, search_results)
            else:
                final_answer = self._answer_with_crewai(question, plan.task_type, search_results)
                final_answer = SummaryAgent(self.llm_client, self._log).guard_contradictions(
                    final_answer,
                    search_results=search_results,
                )

            self._log("Supervisor Agent", "Preparing the final answer.", status="done")
            return _append_sources(final_answer, search_results)
        except LLMClientError as exc:
            self._log(
                "Supervisor Agent",
                "The CrewAI or LLM call failed, so I returned a clear setup message.",
                status="error",
            )
            return str(exc)
        except Exception as exc:
            self._log("Supervisor Agent", "An unexpected app error occurred while answering.", status="error")
            return f"DevVault AI hit an error while answering: {exc}"

    def _make_plan(self, question: str):
        return self._timed_step(
            "Task Agent",
            "Classifying the question before building the CrewAI tasks.",
            lambda: TaskAgent(self._log).plan(question),
        )

    def _search(self, question: str) -> list[dict]:
        return self._timed_step(
            "Search Agent",
            "Searching the vector database for relevant context.",
            lambda: SearchAgent(self.vector_store, self._log).search(question),
        )

    def _answer_with_crewai(self, question: str, task_type: str, search_results: list[dict]) -> str:
        if not search_results:
            return _not_found_answer()

        return self._timed_step(
            "CrewAI Supervisor",
            "Starting the CrewAI answer crew.",
            lambda: CrewAIRagPipeline(self.llm_client, self._log).answer(
                question=question,
                task_type=task_type,
                search_results=search_results,
                conversation_memory=self.conversation_memory,
                agent_mode=self.agent_mode,
            ),
        )

    def _answer_with_document_summary(self, question: str, search_results: list[dict]) -> str:
        document_result = self._timed_step(
            "Document Agent",
            "Preparing a structured document answer from retrieved chunks.",
            lambda: DocumentAgent(self.llm_client, self._log).answer_structured(
                question,
                search_results,
                conversation_memory=self.conversation_memory,
            ),
        )

        return self._timed_step(
            "Summary Agent",
            "Resolving document coverage into one coherent final answer.",
            lambda: SummaryAgent(self.llm_client, self._log).synthesize_document_answer(
                question,
                document_result,
                search_results,
            ),
        )

    def _timed_step(self, agent_name: str, message: str, callback):
        start = time.perf_counter()
        self._log(agent_name, message, status="running")
        try:
            result = callback()
        except Exception:
            elapsed = time.perf_counter() - start
            self._log(agent_name, "Step failed.", status="error", elapsed_seconds=elapsed)
            raise

        elapsed = time.perf_counter() - start
        self._log(agent_name, "Step completed.", status="done", elapsed_seconds=elapsed)
        return result

    def _log(
        self,
        agent_name: str,
        message: str,
        status: str = "info",
        elapsed_seconds: Optional[float] = None,
    ) -> None:
        self.agent_log.append(
            {
                "agent": agent_name,
                "message": message,
                "status": status,
                "elapsed_seconds": elapsed_seconds,
                "time": datetime.now().strftime("%H:%M:%S"),
            }
        )


def _not_found_answer() -> str:
    return (
        "I could not find that information in the uploaded documents or indexed code. "
        "Index the relevant files first, or ask about material that is already in the vector database."
    )


def _append_sources(answer: str, search_results: list[dict]) -> str:
    source_footer = format_source_footer(search_results)
    if not source_footer:
        return answer

    if "sources:" in answer.lower():
        return answer

    return f"{answer.rstrip()}\n\n{source_footer}"


def _should_use_simple_document_route(
    question: str,
    task_type: str,
    search_results: list[dict],
    agent_mode: str,
) -> bool:
    if agent_mode == "debate" or not search_results:
        return False

    if task_type == "code" or not _has_document_context(search_results):
        return False

    return _is_simple_explanation_question(question) and not _asks_for_complex_reasoning(question)


def _has_document_context(search_results: list[dict]) -> bool:
    return any((item.get("metadata") or {}).get("kind") == "document" for item in search_results)


def _is_simple_explanation_question(question: str) -> bool:
    lowered = question.lower()
    words = [word for word in lowered.replace("?", " ").split() if word]
    if len(words) > 16:
        return False

    markers = (
        "what is",
        "what's",
        "what is it",
        "define",
        "definition",
        "meaning",
        "explain",
        "what does",
        "who is",
        "when is",
        "where is",
        "что это",
        "что такое",
        "что значит",
        "объясни",
        "מה זה",
        "מהו",
        "מהי",
        "מה הם",
    )
    return any(marker in lowered for marker in markers)


def _asks_for_complex_reasoning(question: str) -> bool:
    lowered = question.lower()
    markers = (
        "compare",
        "contrast",
        "pros and cons",
        "debate",
        "critic",
        "critique",
        "review",
        "across documents",
        "all documents",
        "architecture",
    )
    return any(marker in lowered for marker in markers)
