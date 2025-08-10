from pydantic_ai import (
    Agent,
)

from pydantic_ai.output import OutputDataT
from typing import Any, Generic
from collections.abc import Callable
from pydantic_ai.tools import AgentDepsT

from streamlit_pydantic_ai_copilot.agent_conversation import AgentConversation
from streamlit_pydantic_ai_copilot.agent_events import (
    ToolRenderer,
)
from streamlit_pydantic_ai_copilot.tool_renderer import AgentResultRenderer
from streamlit_pydantic_ai_copilot.utils import streamlit_session_id


class AgentSession(Generic[AgentDepsT, OutputDataT]):
    agent: Agent[AgentDepsT, OutputDataT]
    tool_renderers: list[ToolRenderer[Any]] | None = None
    result_renderer: AgentResultRenderer[OutputDataT] | None = None
    on_deps_change: Callable[[AgentDepsT], None] | None = None
    _conversations: dict[str, AgentConversation[AgentDepsT, OutputDataT]] = {}

    def __init__(
        self,
        agent: Agent[AgentDepsT, OutputDataT],
        tool_renderers: list[ToolRenderer[Any]] | None = None,
        result_renderer: AgentResultRenderer[OutputDataT] | None = None,
        on_deps_change: Callable[[AgentDepsT], None] | None = None,
    ):
        super().__init__()
        self.agent = agent
        self.count = 0

        self.tool_renderers = tool_renderers
        self.result_renderer = result_renderer
        self.on_deps_change = on_deps_change

    def get_or_create_conversation(
        self, conversation_id: str | None = None
    ) -> AgentConversation[AgentDepsT, OutputDataT]:
        if conversation_id is None:
            conversation_id = streamlit_session_id()

        if conversation_id not in self._conversations:
            self._conversations[conversation_id] = AgentConversation(
                conversation_id=conversation_id,
                agent=self.agent,
                tool_renderers=self.tool_renderers,
                result_renderer=self.result_renderer,
                on_deps_change=self.on_deps_change,
            )
        return self._conversations[conversation_id]
