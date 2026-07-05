from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Optional

from agents.code_agent import INTENT_INSTRUCTIONS, detect_code_intent
from core.llm_client import LLMClientError, friendly_provider_error, is_gpt5_model, normalize_openai_model_name
from core.prompts import CODE_AGENT_PROMPT, DOCUMENT_AGENT_PROMPT, SUPERVISOR_PROMPT
from tools.search_tools import format_search_context


CREWAI_PROVIDER_PREFIX = "openai/"
CREWAI_INSTALL_MESSAGE = (
    "CrewAI is required for this version of DevVault AI. Recreate the environment with Python 3.10-3.13, "
    "then run `python -m pip install -r requirements.txt`."
)


@dataclass(frozen=True)
class CrewAIImports:
    Agent: Any
    Crew: Any
    LLM: Any
    Process: Any
    Task: Any


class CrewAIRagPipeline:
    def __init__(self, llm_client, logger: Optional[Callable] = None):
        self.llm_client = llm_client
        self.logger = logger

    def answer(
        self,
        question: str,
        task_type: str,
        search_results: list[dict],
        conversation_memory: str = "",
        agent_mode: str = "standard",
    ) -> str:
        if not search_results:
            return (
                "I could not find that information in the uploaded documents or indexed code. "
                "Index the relevant files first, or ask about material that is already in the vector database."
            )

        crewai = _load_crewai()
        llm = _create_crewai_llm(crewai.LLM, self.llm_client)
        context = _source_context(task_type, search_results)

        answer_agent = _answer_agent(crewai.Agent, llm, task_type)
        critic_agent = _critic_agent(crewai.Agent, llm, agent_mode)
        supervisor_agent = _supervisor_agent(crewai.Agent, llm)

        answer_task = crewai.Task(
            description=_answer_task_description(
                question=question,
                task_type=task_type,
                context=context,
                conversation_memory=conversation_memory,
            ),
            expected_output=_answer_expected_output(task_type),
            agent=answer_agent,
        )
        critique_task = crewai.Task(
            description=_critique_task_description(
                question=question,
                task_type=task_type,
                context=context,
                agent_mode=agent_mode,
            ),
            expected_output="A short critique listing unsupported claims, missing context, and fixes. Say when no major issue is found.",
            agent=critic_agent,
            context=[answer_task],
        )
        final_task = crewai.Task(
            description=_final_task_description(
                question=question,
                context=context,
                agent_mode=agent_mode,
            ),
            expected_output="A concise final answer grounded only in the retrieved context.",
            agent=supervisor_agent,
            context=[answer_task, critique_task],
        )

        crew = crewai.Crew(
            agents=[answer_agent, critic_agent, supervisor_agent],
            tasks=[answer_task, critique_task, final_task],
            process=crewai.Process.sequential,
            memory=False,
            verbose=False,
        )

        if self.logger:
            self.logger(
                "CrewAI Crew",
                "Running a sequential CrewAI workflow: answer task, critique task, final synthesis task.",
                status="running",
            )

        try:
            result = crew.kickoff()
        except Exception as exc:
            raise LLMClientError(f"CrewAI execution failed: {friendly_provider_error(exc, self.llm_client.config)}") from exc

        if self.logger:
            self.logger("CrewAI Crew", "CrewAI workflow completed.", status="done")

        return _crew_output_text(result)


def _load_crewai() -> CrewAIImports:
    try:
        from crewai import Agent, Crew, LLM, Process, Task
    except Exception as exc:
        raise LLMClientError(f"{CREWAI_INSTALL_MESSAGE} Original import error: {exc}") from exc

    return CrewAIImports(Agent=Agent, Crew=Crew, LLM=LLM, Process=Process, Task=Task)


def _create_crewai_llm(llm_class, llm_client):
    config = llm_client.config

    if config.provider == "openai":
        if not config.openai_api_key:
            raise LLMClientError(
                "OpenAI is selected, but `[openai].api_key` is missing in `.streamlit/secrets.toml`."
            )
        _clear_local_openai_base_url_env()
        os.environ["OPENAI_API_KEY"] = config.openai_api_key
        model_name = normalize_openai_model_name(config.openai_model)
        api_key = config.openai_api_key
        base_url = None
    elif config.provider == "local":
        api_key = "lm-studio"
        base_url = config.local_base_url
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["OPENAI_BASE_URL"] = base_url
        model_name = llm_client.chat_model_name()
    else:
        raise LLMClientError(f"Unknown provider: {config.provider}")

    llm_args = {
        "model": _crewai_openai_model(model_name),
        "api_key": api_key,
        "timeout": 120,
    }
    if base_url:
        llm_args["base_url"] = base_url
    if is_gpt5_model(model_name):
        os.environ["LITELLM_DROP_PARAMS"] = "true"
        llm_args["drop_params"] = True
        llm_args["additional_drop_params"] = ["stop", "temperature", "max_tokens"]
    elif not _uses_default_temperature_only(config.provider, model_name):
        llm_args["temperature"] = 0.2

    return llm_class(**llm_args)


def _clear_local_openai_base_url_env() -> None:
    for env_name in (
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
        "OPENAI_API_BASE_URL",
        "LITELLM_API_BASE",
    ):
        os.environ.pop(env_name, None)


def _crewai_openai_model(model_name: str) -> str:
    model_name = normalize_openai_model_name(model_name)
    return f"{CREWAI_PROVIDER_PREFIX}{model_name}"


def _uses_default_temperature_only(provider: str, model_name: str) -> bool:
    return provider == "openai" and is_gpt5_model(model_name)


def _answer_agent(agent_class, llm, task_type: str):
    if task_type == "code":
        role = "Code Analysis Agent"
        goal = "Explain, debug, refactor, and review code using only retrieved code chunks."
        backstory = CODE_AGENT_PROMPT
    elif task_type == "document":
        role = "Document Answer Agent"
        goal = "Answer document questions using only retrieved document chunks."
        backstory = DOCUMENT_AGENT_PROMPT
    else:
        role = "RAG Answer Agent"
        goal = "Answer project questions using the retrieved source context."
        backstory = SUPERVISOR_PROMPT

    return agent_class(
        role=role,
        goal=goal,
        backstory=backstory,
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )


def _critic_agent(agent_class, llm, agent_mode: str):
    mode_goal = (
        "Challenge the draft like an adversarial reviewer and identify every unsupported claim."
        if agent_mode == "debate"
        else "Check the draft for grounding, missing context, and overconfident claims."
    )
    return agent_class(
        role="Critic Agent",
        goal=mode_goal,
        backstory=(
            "You review answers against retrieved RAG context. You are strict about source grounding, "
            "but you keep feedback short and practical."
        ),
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )


def _supervisor_agent(agent_class, llm):
    return agent_class(
        role="Supervisor Synthesis Agent",
        goal="Resolve the draft and critique into one concise, grounded final answer.",
        backstory=(
            "You coordinate a CrewAI answer workflow for DevVault AI. You preserve useful draft content, "
            "apply valid critique, and clearly state when the retrieved context is incomplete."
        ),
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )


def _answer_task_description(
    question: str,
    task_type: str,
    context: str,
    conversation_memory: str,
) -> str:
    memory = _memory_section(conversation_memory)
    if task_type == "code":
        intent = detect_code_intent(question)
        format_instruction = INTENT_INSTRUCTIONS[intent]
        task_label = f"Code question type: {intent}"
    elif task_type == "document":
        format_instruction = "Answer concisely in plain language and mention source names when useful."
        task_label = "Document question"
    else:
        format_instruction = "Give a clear answer in short sections or bullets when useful."
        task_label = "General RAG question"

    return (
        f"{task_label}\n\n"
        f"{memory}"
        f"Question:\n{question}\n\n"
        f"Required answer format or style:\n{format_instruction}\n\n"
        f"Retrieved context:\n{context}\n\n"
        "Use only the retrieved context as source evidence. If something important is missing, say exactly what is missing. "
        "Do not invent files, sources, code behavior, or document content."
    )


def _critique_task_description(
    question: str,
    task_type: str,
    context: str,
    agent_mode: str,
) -> str:
    style = (
        "This is CrewAI debate mode, so be adversarial but constructive."
        if agent_mode == "debate"
        else "This is standard mode, so keep the review brief."
    )
    return (
        f"{style}\n\n"
        f"Question:\n{question}\n\n"
        f"Task type: {task_type}\n\n"
        f"Retrieved context for checking:\n{context}\n\n"
        "Review the previous task output. Identify unsupported claims, missing source context, weak reasoning, "
        "or places where the answer should be less certain. Do not add new facts from outside the retrieved context."
    )


def _final_task_description(question: str, context: str, agent_mode: str) -> str:
    mode_instruction = (
        "Resolve the debate by keeping useful draft content and applying valid criticism."
        if agent_mode == "debate"
        else "Use the critique as a quality check before returning the final answer."
    )
    return (
        f"{mode_instruction}\n\n"
        f"Question:\n{question}\n\n"
        f"Retrieved context:\n{context}\n\n"
        "You are the final synthesis agent. Your job is not to concatenate previous agent outputs.\n"
        "Before writing the final answer, check whether the previous agents contradict each other.\n"
        "If one agent says the context does not contain information but another agent provides evidence from retrieved chunks, resolve the contradiction.\n"
        "Rules:\n"
        "- If retrieved chunks contain relevant information, do not say that the documents contain no information.\n"
        "- If retrieved chunks contain only partial information, say that the documents contain a partial mention, but not a full definition.\n"
        "- If retrieved chunks contain no relevant information, say that clearly.\n"
        "- Do not include two incompatible claims in the same final answer.\n"
        "- Prefer evidence from retrieved chunks over unsupported agent statements.\n\n"
        "Produce the final answer. Keep it concise, grounded, and practical. "
        "Do not include a Sources section; the application appends source notes after the CrewAI run."
    )


def _answer_expected_output(task_type: str) -> str:
    if task_type == "code":
        return "A structured code answer using the requested code-question headings."
    if task_type == "document":
        return "A concise answer grounded in the retrieved document chunks."
    return "A concise answer grounded in the retrieved source chunks."


def _source_context(task_type: str, search_results: list[dict]) -> str:
    source_label = "File" if task_type == "code" else "Source"
    return format_search_context(search_results, source_label=source_label)


def _memory_section(conversation_memory: str) -> str:
    if not conversation_memory:
        return ""

    return (
        "Conversation memory for follow-up context only, not as source evidence:\n"
        f"{conversation_memory}\n\n"
    )


def _crew_output_text(result) -> str:
    raw = getattr(result, "raw", None)
    if raw:
        return str(raw).strip()

    return str(result).strip()
