from __future__ import annotations


UNCERTAINTY_MARKERS = (
    "not enough",
    "missing",
    "uncertain",
    "cannot tell",
    "not shown",
    "not provided",
    "could not find",
    "incomplete",
)

GENERIC_MARKERS = (
    "it depends",
    "check the code",
    "try debugging",
    "make it cleaner",
    "improve the code",
)

CONFIDENT_MARKERS = (
    "definitely",
    "clearly",
    "the cause is",
    "this happens because",
    "this will fix",
)


class CriticAgent:
    def __init__(self, logger=None):
        self.logger = logger

    def review(self, answer: str, search_results: list[dict]) -> str:
        notes = _find_review_notes(answer, search_results)
        answer_text = answer.strip()

        if not answer_text:
            answer = "I could not produce an answer from the available context."

        if self.logger:
            message = (
                "Checked the answer and found uncertainty to mention."
                if notes
                else "Checked the answer. Sources are present."
            )
            self.logger("Critic Agent", message)

        if not notes:
            return answer

        return answer.rstrip() + "\n\nNote: " + " ".join(notes)

    def debate_feedback(self, answer: str, search_results: list[dict]) -> str:
        notes = _find_review_notes(answer, search_results)

        if self.logger:
            message = (
                "Prepared critique for the debate step."
                if notes
                else "Prepared critique. No major grounding issues found."
            )
            self.logger("Critic Agent", message)

        if not notes:
            return (
                "No major grounding issues found. Keep the final answer tied to the retrieved sources "
                "and mention any missing context briefly."
            )

        return " ".join(notes)


def _find_review_notes(answer: str, search_results: list[dict]) -> list[str]:
    notes = []
    answer_text = answer.strip()

    if not answer_text:
        notes.append("The LLM returned an empty answer.")

    if not search_results and "could not find" not in answer.lower():
        notes.append("No RAG sources were found, so the answer may be based only on the model.")

    if _is_too_generic(answer_text):
        notes.append("The answer is very general; a specific file, function, or error message may be needed.")

    if _needs_missing_context_note(answer_text, search_results):
        notes.append("Only limited code context was retrieved, so concrete claims should be treated as tentative.")

    return notes


def _is_too_generic(answer: str) -> bool:
    lowered = answer.lower()
    return len(answer) < 300 and any(marker in lowered for marker in GENERIC_MARKERS)


def _needs_missing_context_note(answer: str, search_results: list[dict]) -> bool:
    if len(search_results) >= 2:
        return False

    lowered = answer.lower()
    mentions_uncertainty = any(marker in lowered for marker in UNCERTAINTY_MARKERS)
    sounds_confident = any(marker in lowered for marker in CONFIDENT_MARKERS)
    return sounds_confident and not mentions_uncertainty
