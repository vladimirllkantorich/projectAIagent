from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class TaskPlan:
    task_type: str
    agents: list[str]
    reason: str


class TaskAgent:
    CODE_WORDS = {
        "code",
        "function",
        "class",
        "bug",
        "refactor",
        "architecture",
        "dependency",
        "error",
        "import",
        "repository",
        "project",
        "folder",
        "module",
        "path",
        "rag",
        "streamlit",
        "codeagent",
        "supervisoragent",
        "searchagent",
        "vectorstore",
        "agent",
        "agents",
        "traceback",
        "exception",
        "crash",
        "pipeline",
    }
    DOCUMENT_WORDS = {
        "document",
        "pdf",
        "docx",
        "file",
        "source",
        "summarize",
        "summary",
        "notes",
        "text",
        "article",
        "chunk",
        "page",
    }

    def __init__(self, logger=None):
        self.logger = logger

    def plan(self, question: str) -> TaskPlan:
        task_type = self._task_type_for_question(question)
        plan = _plan_for_task_type(task_type)

        if self.logger:
            self.logger("Task Agent", f"{plan.reason} Route: {', '.join(plan.agents)}.")

        return plan

    def _task_type_for_question(self, question: str) -> str:
        words = set(re.findall(r"[a-z0-9_]+", question.lower()))

        if words & self.CODE_WORDS:
            return "code"
        if words & self.DOCUMENT_WORDS:
            return "document"
        return "general"


def _plan_for_task_type(task_type: str) -> TaskPlan:
    if task_type == "code":
        return TaskPlan(
            task_type="code",
            agents=["Search Agent", "Code Agent", "Critic Agent", "Summary Agent"],
            reason="The question appears to be about code or project structure.",
        )

    if task_type == "document":
        return TaskPlan(
            task_type="document",
            agents=["Search Agent", "Document Agent", "Critic Agent", "Summary Agent"],
            reason="The question appears to be about indexed documents.",
        )

    return TaskPlan(
        task_type="general",
        agents=["Search Agent", "Supervisor Agent", "Critic Agent", "Summary Agent"],
        reason="The question is general, so the supervisor will answer using any matching context.",
    )
