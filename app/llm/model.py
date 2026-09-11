"""Provider-Agnostic LLM Factory Module for DataPilot."""

import logging
from typing import Optional, List, Any, Dict
from pydantic import Field
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from app.core.config import settings

logger = logging.getLogger("datapilot.llm")


class MockChatModel(BaseChatModel):
    """Mock ChatModel for offline unit testing without live API keys."""
    
    mock_response: str = "DataPilot Mock LLM Response: Analysis successfully processed."

    @property
    def _llm_type(self) -> str:
        return "mock-chat-model"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        ai_msg = AIMessage(content=self.mock_response)
        generation = ChatGeneration(message=ai_msg)
        return ChatResult(generations=[generation])


def get_llm_model(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
) -> BaseChatModel:
    """Provider-agnostic factory function returning a initialized LangChain BaseChatModel instance.

    Configurable via LLM_PROVIDER, LLM_MODEL_NAME, and LLM_TEMPERATURE.
    Supported providers: 'openai', 'google' ('gemini'), 'anthropic', 'mock'.
    """
    target_provider = (provider or settings.LLM_PROVIDER).lower()
    target_model = model_name or settings.LLM_MODEL_NAME
    target_temp = temperature if temperature is not None else settings.LLM_TEMPERATURE

    if target_provider == "mock":
        return MockChatModel()

    elif target_provider == "openai":
        api_key = settings.OPENAI_API_KEY
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set in environment settings.")
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=target_model,
            temperature=target_temp,
            api_key=api_key,
        )

    elif target_provider in ["google", "gemini"]:
        api_key = settings.GOOGLE_API_KEY
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is not set in environment settings.")
        from langchain_google_genai import ChatGoogleGenerativeAI
        actual_model = target_model if target_model != "gpt-4o" else "gemini-1.5-pro"
        return ChatGoogleGenerativeAI(
            model=actual_model,
            temperature=target_temp,
            google_api_key=api_key,
        )

    elif target_provider == "anthropic":
        api_key = settings.ANTHROPIC_API_KEY
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set in environment settings.")
        from langchain_anthropic import ChatAnthropic
        actual_model = target_model if target_model != "gpt-4o" else "claude-3-5-sonnet-20240620"
        return ChatAnthropic(
            model=actual_model,
            temperature=target_temp,
            api_key=api_key,
        )

    else:
        raise ValueError(f"Unsupported LLM provider '{target_provider}'. Supported: 'openai', 'google', 'anthropic', 'mock'.")
