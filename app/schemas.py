"""OpenAI-compatible request and response models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    field_validator,
    model_validator,
)


class ToolCallFunctionIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    arguments: str


class ToolCallIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    type: Literal["function"] = "function"
    function: ToolCallFunctionIn


class ChatMessageIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[ToolCallIn] | None = None
    tool_call_id: str | None = None

    @model_validator(mode="after")
    def validate_role_fields(self):
        if self.role == "tool":
            if not self.tool_call_id or self.content is None or self.tool_calls:
                raise ValueError("tool messages require content and tool_call_id")
        else:
            if self.tool_call_id is not None:
                raise ValueError("tool_call_id is only valid on tool messages")
            if self.tool_calls and self.role != "assistant":
                raise ValueError("tool_calls are only valid on assistant messages")
            if self.content is None and not (
                self.role == "assistant" and self.tool_calls
            ):
                raise ValueError("content is required")
        return self


class ToolChoiceFunctionIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)


class ToolChoiceIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["function"]
    function: ToolChoiceFunctionIn


class JsonSchemaEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        default="response", min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$"
    )
    schema_: dict[str, Any] | StrictBool = Field(alias="schema")
    description: str | None = None
    strict: StrictBool | None = None


class ResponseFormatIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["json_schema"]
    json_schema: dict[str, Any]

    @model_validator(mode="after")
    def validate_envelope(self):
        if "schema" in self.json_schema:
            JsonSchemaEnvelope.model_validate(self.json_schema)
        return self


class StreamOptionsIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    include_usage: StrictBool = False


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
    stream_options: StreamOptionsIn | None = None
    user: str | None = None
    n: int = 1
    response_format: ResponseFormatIn | None = None
    tools: list[ToolIn] | None = None
    tool_choice: Literal["none", "auto", "required"] | ToolChoiceIn | None = None
    prompt: PromptIn | None = None

    @model_validator(mode="after")
    def validate_stream_options(self):
        if self.stream_options is not None and not self.stream:
            raise ValueError("stream_options requires stream=true")
        return self

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
    content: str | None
    tool_calls: list[ToolCallIn] | None = None
    refusal: str | None = None


class Choice(BaseModel):
    index: int = 0
    message: ChoiceMessage
    finish_reason: str


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    model: str
    choices: list[Choice]
    created: int
    usage: dict[str, Any]


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "gateway"
    created: int = 0


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelCard]
