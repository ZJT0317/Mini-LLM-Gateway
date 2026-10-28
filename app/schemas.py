"""OpenAI-compatible request and response models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChatMessageIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str


class ResponseFormatIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["json_schema"]
    json_schema: dict[str, Any]


class ToolFunctionIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)


class ToolIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["function"]
    function: ToolFunctionIn


class PromptIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    version: str
    variables: dict[str, Any] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = ""
    messages: list[ChatMessageIn] = Field(default_factory=list)
    max_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = None
    stream: bool = False
    user: str | None = None
    n: int = 1
    response_format: ResponseFormatIn | None = None
    tools: list[ToolIn] | None = None
    tool_choice: str | dict | None = None
    prompt: PromptIn | None = None

    @field_validator("n")
    @classmethod
    def n_must_be_one(cls, value: int) -> int:
        if value != 1:
            raise ValueError("n must be 1")
        return value

    @field_validator("temperature")
    @classmethod
    def temperature_range(cls, value: float | None) -> float | None:
        if value is not None and not 0 <= value <= 2:
            raise ValueError("temperature must be between 0 and 2")
        return value


class PromptRenderRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    version: str
    variables: dict[str, Any] = Field(default_factory=dict)


class ChoiceMessage(BaseModel):
    role: str = "assistant"
    content: str


class Choice(BaseModel):
    index: int = 0
    message: ChoiceMessage
    finish_reason: str


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    model: str
    choices: list[Choice]


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "gateway"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelCard]
