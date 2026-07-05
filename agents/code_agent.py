from __future__ import annotations

from core.prompts import CODE_AGENT_PROMPT
from tools.search_tools import format_search_context


INTENT_KEYWORDS = {
    "debug": ("bug", "error", "exception", "crash", "fail", "broken", "traceback"),
    "refactor": ("refactor", "clean", "simplify", "improve", "rewrite", "duplicate"),
    "rag_project": ("rag", "vector", "vectorstore", "chroma", "embedding", "chunk", "retrieval"),
    "architecture": ("architecture", "structure", "pipeline", "design", "organize"),
    "imports": ("import", "dependency", "package"),
}

INTENT_INSTRUCTIONS = {
    "explain": """
Use this answer format:
- What this code does
- Main flow
- Important functions/classes
- Possible problems
- Suggested improvements
""".strip(),
    "debug": """
Use this answer format:
- Likely cause
- Where to check
- Fix
- Why this fix works
""".strip(),
    "refactor": """
Use this answer format:
- What is currently complicated
- What can be simplified
- Suggested cleaner version
- Why it is better
""".strip(),
    "architecture": """
Use this answer format:
- Current structure
- What is good
- What is weak
- Recommended structure
""".strip(),
    "imports": """
Use this answer format:
- Import/dependency issue
- Where to check
- Fix
- Why this fix works
""".strip(),
    "rag_project": """
Use this answer format:
- Current RAG/project flow
- How data moves through the code
- Important modules/classes
- Possible weak spots
- Suggested improvements
""".strip(),
}


class CodeAgent:
    def __init__(self, llm_client, logger=None):
        self.llm_client = llm_client
        self.logger = logger

    def answer(
        self,
        question: str,
        search_results: list[dict],
        conversation_memory: str = "",
    ) -> str:
        if self.logger:
            self.logger("Code Agent", "Analyzing the retrieved code chunks.")

        if not search_results:
            return "I could not find relevant code in the indexed project."

        intent = self._detect_intent(question)
        context = format_search_context(search_results, source_label="File")
        messages = [
            {"role": "system", "content": CODE_AGENT_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Code question type: {intent}\n\n"
                    f"{_memory_section(conversation_memory)}"
                    f"{INTENT_INSTRUCTIONS[intent]}\n\n"
                    f"Question:\n{question}\n\n"
                    f"Relevant code chunks:\n{context}\n\n"
                    "Use only these chunks as the main evidence. "
                    "If something important is missing, say exactly what is missing."
                ),
            },
        ]
        return self.llm_client.chat(messages)

    def _detect_intent(self, question: str) -> str:
        return detect_code_intent(question)


def detect_code_intent(question: str) -> str:
    text = question.lower()

    for intent, keywords in INTENT_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            return intent

    return "explain"


def _memory_section(conversation_memory: str) -> str:
    if not conversation_memory:
        return ""

    return (
        "Conversation memory for follow-up context only, not as code evidence:\n"
        f"{conversation_memory}\n\n"
    )
