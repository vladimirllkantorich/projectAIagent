from __future__ import annotations

import os

from openai import OpenAI

from core.config import AppConfig


LOCAL_MODEL_PLACEHOLDERS = {"", "auto", "local-model"}
OPENAI_PROVIDER_PREFIX = "openai/"
OPENAI_AUTH_MARKERS = ("api key", "authentication", "unauthorized")
LOCAL_CONNECTION_MARKERS = (
    "connection refused",
    "failed to establish a new connection",
    "max retries exceeded",
    "winerror 10061",
    "could not connect",
    "connecterror",
)


class LLMClientError(RuntimeError):
    """Raised when DevVault AI cannot complete a chat request."""


class LLMClient:
    def __init__(self, config: AppConfig):
        self.config = config

    def validate_provider(self) -> None:
        if self.config.provider == "openai":
            _validate_openai_config(self.config)
            return

        if self.config.provider == "local":
            self.chat_model_name()
            return

        raise LLMClientError(f"Unknown provider: {self.config.provider}")

    def chat_model_name(self) -> str:
        if self.config.provider == "openai":
            return normalize_openai_model_name(self.config.openai_model)

        if self.config.provider == "local":
            client = OpenAI(base_url=self.config.local_base_url, api_key="lm-studio")
            return self._resolve_local_chat_model(client)

        raise LLMClientError(f"Unknown provider: {self.config.provider}")

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.2) -> str:
        if self.config.provider == "openai":
            return self._chat_openai(messages, temperature)

        if self.config.provider == "local":
            return self._chat_local(messages, temperature)

        raise LLMClientError(f"Unknown provider: {self.config.provider}")

    def _chat_openai(self, messages: list[dict[str, str]], temperature: float) -> str:
        if not self.config.openai_api_key:
            raise LLMClientError(
                "OpenAI is selected, but `[openai].api_key` is missing in `.streamlit/secrets.toml`."
            )

        _clear_openai_base_url_env()
        client = OpenAI(api_key=self.config.openai_api_key)
        return self._send_chat_request(
            client=client,
            model=normalize_openai_model_name(self.config.openai_model),
            messages=messages,
            temperature=temperature,
            provider="openai",
        )

    def _chat_local(self, messages: list[dict[str, str]], temperature: float) -> str:
        client = OpenAI(base_url=self.config.local_base_url, api_key="lm-studio")
        model = self._resolve_local_chat_model(client)
        return self._send_chat_request(
            client=client,
            model=model,
            messages=messages,
            temperature=temperature,
            provider="local",
        )

    def _resolve_local_chat_model(self, client: OpenAI) -> str:
        configured_model = (self.config.local_model or "").strip()
        chat_model_ids = _local_chat_model_ids(client, self.config.local_base_url)

        if configured_model.lower() not in LOCAL_MODEL_PLACEHOLDERS:
            if configured_model not in chat_model_ids:
                loaded = ", ".join(chat_model_ids) or "none"
                raise LLMClientError(
                    f"LM Studio model `{configured_model}` is not currently loaded. "
                    f"Loaded chat models: {loaded}. Load the configured model, "
                    "or set `LM_STUDIO_MODEL=local-model` / `[local].model = \"local-model\"` to auto-select one."
                )
            return configured_model

        return chat_model_ids[0]

    def _send_chat_request(
        self,
        client: OpenAI,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        provider: str,
    ) -> str:
        request_args = _chat_request_args(provider, model, messages, temperature)
        try:
            response = client.chat.completions.create(**request_args)
        except Exception as exc:
            raise LLMClientError(
                _friendly_llm_error(
                    exc=exc,
                    model=model,
                    provider=provider,
                    local_base_url=self.config.local_base_url,
                )
            ) from exc

        content = response.choices[0].message.content
        return content.strip() if content else ""


def _friendly_llm_error(
    exc: Exception,
    model: str,
    provider: str,
    local_base_url: str,
) -> str:
    message = str(exc)
    lowered = message.lower()

    if provider == "local" and _looks_like_connection_error(lowered):
        return (
            "LM Studio is not reachable. Open LM Studio, load a chat model, start the local server, "
            f"and confirm it is listening at `{local_base_url}`."
        )

    if "no models loaded" in lowered:
        return _no_lm_studio_model_message()

    if provider == "local" and "model" in lowered and ("not found" in lowered or "not loaded" in lowered):
        return (
            f"LM Studio could not use chat model `{model}`. Load that model in LM Studio, "
            "or set `[local].model = \"auto\"` in `.streamlit/secrets.toml` to use the first loaded model."
        )

    if provider == "openai" and _contains_any(lowered, OPENAI_AUTH_MARKERS):
        return "OpenAI rejected the request. Check `[openai].api_key` in `.streamlit/secrets.toml`."

    if provider == "openai" and is_gpt5_model(model) and "unsupported" in lowered and "parameter" in lowered:
        return (
            f"OpenAI model `{normalize_openai_model_name(model)}` rejected an unsupported request parameter. "
            "DevVault AI now omits GPT-5 temperature settings; restart the Streamlit app so the updated code is loaded."
        )

    return f"LLM request failed while using model `{model}`: {exc}"


def friendly_provider_error(exc: Exception, config: AppConfig) -> str:
    model = normalize_openai_model_name(config.openai_model) if config.provider == "openai" else config.local_model
    return _friendly_llm_error(
        exc=exc,
        model=model,
        provider=config.provider,
        local_base_url=config.local_base_url,
    )


def _validate_openai_config(config: AppConfig) -> None:
    if not config.openai_api_key:
        raise LLMClientError(
            "OpenAI is selected, but no API key was found. Add `[openai] api_key = \"sk-...\"` "
            "to the real `.streamlit/secrets.toml` file."
        )

    if not (config.openai_model or "").strip():
        raise LLMClientError(
            "OpenAI is selected, but the model name is empty. Set `OPENAI_MODEL=gpt-5-mini` "
            "or `[openai].model = \"gpt-5-mini\"`."
        )

    model_name = normalize_openai_model_name(config.openai_model)
    if model_name.lower() in LOCAL_MODEL_PLACEHOLDERS:
        raise LLMClientError(
            f"OpenAI model is set to `{config.openai_model}`, which is an LM Studio placeholder. "
            "Set `OPENAI_MODEL=gpt-5-mini` or another OpenAI chat model."
        )


def _local_chat_model_ids(client: OpenAI, local_base_url: str) -> list[str]:
    try:
        models = client.models.list()
        model_ids = [model.id for model in models.data if getattr(model, "id", None)]
    except Exception as exc:
        raise LLMClientError(
            "Could not read loaded models from LM Studio. Make sure the LM Studio server is running "
            f"at `{local_base_url}`. Original error: {exc}"
        ) from exc

    chat_model_ids = [model_id for model_id in model_ids if not _looks_like_embedding_model(model_id)]
    if not chat_model_ids:
        raise LLMClientError(_no_lm_studio_model_message())

    return chat_model_ids


def _no_lm_studio_model_message() -> str:
    return (
        "No model is currently loaded in LM Studio.\n"
        "Open LM Studio, load a model, start the local server, and try again.\n"
        "You can also use the `lms load` command if LM Studio CLI is installed."
    )


def _chat_request_args(
    provider: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
) -> dict:
    request_args = {
        "model": normalize_openai_model_name(model) if provider == "openai" else model,
        "messages": messages,
    }
    if not _uses_default_temperature_only(provider, model):
        request_args["temperature"] = temperature
    return request_args


def _uses_default_temperature_only(provider: str, model: str) -> bool:
    if provider != "openai":
        return False

    return is_gpt5_model(model)


def normalize_openai_model_name(model: str) -> str:
    model_name = (model or "").strip()
    if model_name.lower().startswith(OPENAI_PROVIDER_PREFIX):
        return model_name[len(OPENAI_PROVIDER_PREFIX) :]
    return model_name


def is_gpt5_model(model: str) -> bool:
    return normalize_openai_model_name(model).lower().startswith("gpt-5")


def _clear_openai_base_url_env() -> None:
    for env_name in (
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
        "OPENAI_API_BASE_URL",
        "LITELLM_API_BASE",
    ):
        os.environ.pop(env_name, None)


def _looks_like_embedding_model(model_id: str) -> bool:
    lowered = model_id.lower()
    return "embed" in lowered or "embedding" in lowered


def _looks_like_connection_error(lowered_message: str) -> bool:
    return _contains_any(lowered_message, LOCAL_CONNECTION_MARKERS)


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)
