from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from typing import Any, Generic, Literal

from pydantic_ai.agent import AgentRunResult
from pydantic_ai.output import OutputDataT
import streamlit as st
from typing import Callable

from streamlit_pydantic_ai_copilot.tool_renderer import (
    AgentResultRenderer,
    StreamlitToolDefinition,
    ToolResult,
    ToolOuputT,
    ToolRenderer,
    ToolStatus,
)


logger = logging.getLogger(__name__)


class BaseEvent(ABC):
    @abstractmethod
    def render(self) -> None:
        raise NotImplementedError


@dataclass
class ChatMessageEvent(BaseEvent):
    content: str
    role: Literal["user", "assistant"]

    def render(self) -> None:
        ##TODO: make this configurable
        if self.role == "user":
            st.chat_message("user").write(self.content)
        else:
            st.chat_message("assistant").write(self.content)


@dataclass
class UserMessageEvent(ChatMessageEvent):
    def __init__(self, content: str):
        super().__init__(role="user", content=content)


@dataclass
class AssistantMessageEvent(ChatMessageEvent):
    def __init__(self, content: str):
        super().__init__(role="assistant", content=content)


@dataclass
class ToolCallEvent(BaseEvent, Generic[ToolOuputT]):
    id: str
    name: str
    args: dict[str, Any]
    result: ToolResult[ToolOuputT]
    renderer: ToolRenderer[ToolOuputT] | None = None

    def render(self) -> None:
        if self.renderer:
            self.renderer.render(self.args, self.result)


@dataclass
class StreamlitToolCallEvent(BaseEvent, Generic[ToolOuputT]):
    id: str
    name: str
    args: dict[str, Any]
    tool: StreamlitToolDefinition[ToolOuputT]
    on_complete: Callable[["StreamlitToolCallEvent[ToolOuputT]"], None]
    result: Any | None = None

    @property
    def is_completed(self) -> bool:
        return bool(self.result)

    def render(self):
        if not self.is_completed:
            self.result = self._call_handle()
            if self.result:
                self.on_complete(self)
                st.rerun()

        else:
            self._call_render()

    def _call_handle(self) -> Any | None:
        args, kwargs = self.tool.schema._call_args(self.args, None)  # pyright: ignore[reportPrivateUsage, reportArgumentType]
        return self.tool.handle(*args, **kwargs)

    def _call_render(self) -> None:
        tool_result = ToolResult[ToolOuputT](status=ToolStatus.COMPLETED, output=self.result)
        self.tool.render(self.args, tool_result)


@dataclass
class AgentRunResultEvent(BaseEvent, Generic[OutputDataT]):
    result: AgentRunResult[OutputDataT]
    result_renderer: AgentResultRenderer[OutputDataT] | None = None
    role: Literal["user", "assistant"] = "assistant"

    def render(self) -> None:
        if self.result_renderer:
            self.result_renderer.render(self.result)


type AgentEvent[OutputDataT] = (
    UserMessageEvent
    | AssistantMessageEvent
    | ToolCallEvent[Any]
    | StreamlitToolCallEvent[Any]
    | AgentRunResultEvent[OutputDataT]
)
