from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig, ensure_config
from langchain_openai.chat_models.base import BaseChatOpenAI

@dataclass(kw_only=True)
class Configuration:
    model: str = field(
        default="ds/deepseek-chat",
        metadata={
            "description": "The name of the language model to use for the agent's main interactions. "
            "Should be in the form: provider/model-name."
        },
    )

    base_url: Optional[str] = field(
        default=None,
        metadata={
            "description": "The base URL for the model API."
        },
    )

    api_key: str = field(
        default="sk-...",
        metadata={
            "description": "The API key to use for the model API."
        },
    )

    def get_chat_model(self) -> BaseChatModel:
        # https://github.com/langchain-ai/langchain/issues/29282
        return BaseChatOpenAI(model=self.model, openai_api_base=self.base_url, openai_api_key=self.api_key)

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> Configuration:
        """Create a Configuration instance from a RunnableConfig object."""
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})
