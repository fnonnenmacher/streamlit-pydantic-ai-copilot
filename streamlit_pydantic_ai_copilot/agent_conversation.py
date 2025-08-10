from abc import ABC, abstractmethod
import asyncio
import copy
from datetime import datetime
from enum import Enum
from threading import Thread
import time
from typing import Any, Generic
from collections.abc import Callable, Generator
from pydantic_ai import Agent, CallToolsNode, ModelRequestNode, UserPromptNode
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.messages import ModelMessage, ModelRequest, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai.output import DeferredToolCalls, OutputDataT, OutputSpec
from pydantic_ai.tools import AgentDepsT, ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, DeferredToolset
from streamlit_pydantic_ai_copilot.agent_events import (
    AgentEvent,
    AgentRunResultEvent,
    AssistantMessageEvent,
    StreamlitToolCallEvent,
    ToolCallEvent,
    UserMessageEvent,
)
from streamlit_pydantic_ai_copilot.tool_renderer import (
    AgentResultRenderer,
    StreamlitToolDefinition,
    ToolRenderer,
    ToolResult,
    ToolStatus,
)


class EventManager(ABC, Generic[AgentDepsT, OutputDataT]):
    @abstractmethod
    def new_user_message_event(self, part: UserPromptPart) -> UserMessageEvent:
        pass

    @abstractmethod
    def new_assistant_message_event(self, part: TextPart) -> AssistantMessageEvent:
        pass

    @abstractmethod
    def new_tool_call_event(self, part: ToolCallPart) -> ToolCallEvent[Any]:
        pass

    @abstractmethod
    def new_streamlit_tool_call_event(self, part: ToolCallPart) -> StreamlitToolCallEvent[Any]:
        pass

    @abstractmethod
    def new_agent_run_result_event(self, result: AgentRunResult[OutputDataT]) -> AgentRunResultEvent[OutputDataT]:
        pass

    @abstractmethod
    def get_tool_call_event(self, tool_call_id: str) -> ToolCallEvent[Any] | None:
        pass


class AgentDepsChangeDetector(ABC, Generic[AgentDepsT]):
    deps: AgentDepsT | None = None
    _previous_deps: AgentDepsT | None = None
    deps_has_changed: bool = False

    def set_deps(self, deps: AgentDepsT) -> None:
        if deps is None:
            return

        self.deps = deps
        if self._previous_deps is None:
            self._previous_deps = copy.deepcopy(deps)

        # if the deps have changed, set the flag to true and update the previous deps
        # TODO: Is there a better way to detect a change?
        if self._previous_deps != self.deps:
            self.deps_has_changed = True
            self._previous_deps = copy.deepcopy(deps)

    def reset(self) -> None:
        self.deps_has_changed = False


class AgentConversationStatus(Enum):
    INITIALIZED = "initialized"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"


class RunAgentThread(Thread, Generic[AgentDepsT, OutputDataT]):
    agent: Agent[AgentDepsT, OutputDataT]
    event_manager: EventManager[AgentDepsT, OutputDataT]
    streamlit_tool_definitions: dict[str, ToolDefinition]
    user_prompt: str | None
    deps: AgentDepsT
    deps_change_detector: AgentDepsChangeDetector[AgentDepsT]
    message_history: list[ModelMessage] | None = None
    _task: asyncio.Task[AgentRunResult[OutputDataT]] | None = None
    _loop: asyncio.AbstractEventLoop | None = None

    def __init__(
        self,
        agent: Agent[AgentDepsT, OutputDataT],
        event_manager: EventManager[AgentDepsT, OutputDataT],
        streamlit_tool_definitions: dict[str, ToolDefinition],
        deps_change_detector: AgentDepsChangeDetector[AgentDepsT],
        user_prompt: str | None = None,
        deps: AgentDepsT = None,
        message_history: list[ModelMessage] | None = None,
    ):
        super().__init__()
        self.agent = agent
        self.user_prompt = user_prompt
        self.deps = deps
        self.deps_change_detector = deps_change_detector
        self.event_manager = event_manager
        self.message_history = message_history or []
        self.streamlit_tool_definitions = streamlit_tool_definitions

    def run(self):
        self._loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(self._loop)
            self._task = self._loop.create_task(self.async_run())
            self._loop.run_until_complete(self._task)
        finally:
            self._loop.close()

    async def async_run(self) -> AgentRunResult[OutputDataT]:  # noqa: C901 #TODO: refactor this in smaller functions
        toolsets: list[AbstractToolset[AgentDepsT]] = []
        output_type: OutputSpec[OutputDataT] = self.agent.output_type

        streamlit_tools = self.streamlit_tool_definitions.values()
        if streamlit_tools:
            toolset = DeferredToolset[AgentDepsT](list(self.streamlit_tool_definitions.values()))
            toolsets = [toolset]
            output_type = [output_type | DeferredToolCalls]  # pyright: ignore[reportAssignmentType]

        async with self.agent.iter(
            user_prompt=self.user_prompt,
            deps=self.deps,
            toolsets=toolsets,
            output_type=output_type,
            message_history=self.message_history,
        ) as stream:
            async for node in stream:
                # kind of hack we want to detect changes on the agent
                # so we basically check on each new node if they have changed
                self.deps_change_detector.set_deps(stream.ctx.deps.tool_manager.ctx.deps)

                if isinstance(node, UserPromptNode):
                    # skip
                    pass
                elif isinstance(node, ModelRequestNode):
                    for part in node.request.parts:
                        if isinstance(part, UserPromptPart):
                            self.event_manager.new_user_message_event(part)
                        elif isinstance(part, ToolReturnPart):
                            event = self.event_manager.get_tool_call_event(part.tool_call_id)
                            if event:
                                event.result.complete(part.content)
                elif isinstance(node, CallToolsNode):
                    for part in node.model_response.parts:
                        if isinstance(part, ToolCallPart):
                            if part.tool_name in self.streamlit_tool_definitions:
                                self.event_manager.new_streamlit_tool_call_event(part)
                            else:
                                self.event_manager.new_tool_call_event(part)
                        elif isinstance(part, TextPart):
                            self.event_manager.new_assistant_message_event(part)
            assert stream.result is not None
            # TODO: this needs to be done in a better way, streaming of the agent results won't work like this.
            if not isinstance(stream.result.output, DeferredToolCalls):
                self.event_manager.new_agent_run_result_event(stream.result)
        return stream.result

    def stop(self):
        if self._task is not None and not self._task.done():
            self._task.cancel()
        ## TODO do we need to implement a grace period here?
        # Also signal the loop to stop if it's running
        if self._loop and self._loop.is_running():
            self._loop.close()

    def result(self) -> AgentRunResult[OutputDataT]:
        if self._task is None:
            raise ValueError("Conversation not running")
        return self._task.result()


class AgentConversation(EventManager[AgentDepsT, OutputDataT], Generic[AgentDepsT, OutputDataT]):
    conversation_id: str
    creation_time: datetime
    agent: Agent[AgentDepsT, OutputDataT]
    streamlit_tools: dict[str, StreamlitToolDefinition[Any]] = {}
    tool_renderers: dict[str, ToolRenderer[Any]] = {}
    result_renderer: AgentResultRenderer[OutputDataT] | None = None
    on_deps_change: Callable[[AgentDepsT], None] | None = None
    _events: list[AgentEvent[OutputDataT]]
    _run_thread: RunAgentThread[AgentDepsT, OutputDataT] | None = None
    deps_change_detector: AgentDepsChangeDetector[AgentDepsT]

    def __init__(
        self,
        conversation_id: str,
        agent: Agent[AgentDepsT, OutputDataT],
        tool_renderers: list[ToolRenderer[Any]] | None = None,
        result_renderer: AgentResultRenderer[OutputDataT] | None = None,
        on_deps_change: Callable[[AgentDepsT], None] | None = None,
    ):
        super().__init__()
        self.conversation_id = conversation_id
        self.creation_time = datetime.now()
        self.agent = agent
        self._events = []
        self.deps_change_detector = AgentDepsChangeDetector()
        for renderer in tool_renderers or []:
            if isinstance(renderer, StreamlitToolDefinition):
                self.streamlit_tools[renderer.name] = renderer
            else:
                self.tool_renderers[renderer.name] = renderer
        self.result_renderer = result_renderer
        self.on_deps_change = on_deps_change

    def run(
        self,
        user_prompt: str | None = None,
        deps: AgentDepsT = None,
        additional_messages: list[ModelRequest] | None = None,
    ):
        streamlit_tool_definitions = {name: tool.as_tool_definition() for name, tool in self.streamlit_tools.items()}

        message_history = self.last_result.all_messages() if self.last_result else []
        if additional_messages:
            message_history.extend(additional_messages)

        self.deps_change_detector.set_deps(deps)

        self._run_thread = RunAgentThread(
            agent=self.agent,
            event_manager=self,
            streamlit_tool_definitions=streamlit_tool_definitions,
            deps_change_detector=self.deps_change_detector,
            user_prompt=user_prompt,
            deps=deps,
            message_history=message_history,
        )
        self._run_thread.start()

    def stream(
        self,
    ) -> Generator[AgentEvent[OutputDataT], None, None]:
        i = 0
        while self.status == AgentConversationStatus.RUNNING or i < len(self._events):
            if self.on_deps_change and self.deps_change_detector.deps_has_changed:
                self.deps_change_detector.reset()
                deps = self.deps_change_detector.deps
                if deps is not None:
                    self.deps_change_detector.reset()
                    self.on_deps_change(deps)  # type: ignore

            if i < len(self._events):
                yield self._events[i]
                i += 1
            else:
                time.sleep(0.1)

    @property
    def last_result(self) -> AgentRunResult[OutputDataT] | None:
        if self._run_thread is None:
            return None
        return self._run_thread.result()

    def stop(self):
        if self._run_thread is not None:
            self._run_thread.stop()
        pass

    @property
    def status(self) -> AgentConversationStatus:
        if self._run_thread is None:
            return AgentConversationStatus.INITIALIZED
        if self._run_thread.is_alive():
            return AgentConversationStatus.RUNNING
        return AgentConversationStatus.COMPLETED

    # EventFactory methods

    def new_user_message_event(self, part: UserPromptPart) -> UserMessageEvent:
        event = UserMessageEvent(content=str(part.content))
        self._events.append(event)
        return event

    def new_assistant_message_event(self, part: TextPart) -> AssistantMessageEvent:
        event = AssistantMessageEvent(content=part.content)
        self._events.append(event)
        return event

    def new_tool_call_event(self, part: ToolCallPart) -> ToolCallEvent[Any]:
        event = ToolCallEvent(
            id=part.tool_call_id,
            name=part.tool_name,
            args=part.args_as_dict(),
            result=ToolResult(status=ToolStatus.RUNNING),
            renderer=self.tool_renderers.get(part.tool_name, None),
        )
        self._events.append(event)
        return event

    def new_streamlit_tool_call_event(self, part: ToolCallPart) -> StreamlitToolCallEvent[Any]:
        tool = self.streamlit_tools.get(part.tool_name, None)
        if tool is None:
            raise ValueError(f"{part.tool_name} is not a streamlit tool")

        event = StreamlitToolCallEvent(
            id=part.tool_call_id,
            name=part.tool_name,
            args=part.args_as_dict(),
            tool=tool,
            on_complete=self.on_streamlit_tool_call_complete,
        )
        self._events.append(event)
        return event

    def new_agent_run_result_event(self, result: AgentRunResult[OutputDataT]) -> AgentRunResultEvent[OutputDataT]:
        event = AgentRunResultEvent[OutputDataT](result=result, result_renderer=self.result_renderer)
        self._events.append(event)
        return event

    def get_tool_call_event(self, tool_call_id: str) -> ToolCallEvent[Any] | None:
        for event in self._events:
            if isinstance(event, ToolCallEvent) and event.id == tool_call_id:
                return event
        return None

    def on_streamlit_tool_call_complete(self, streamlit_tool_call: StreamlitToolCallEvent[AgentDepsT]) -> None:
        """
        This is a callback that is called when a streamlit tool call is complete.
        It calls the agent with the result of the tool call and ensures that the conversation continues.
        """
        model_request = ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name=streamlit_tool_call.name,
                    tool_call_id=streamlit_tool_call.id,
                    content=streamlit_tool_call.result,
                )
            ]
        )
        self.run(deps=self.deps_change_detector.deps, additional_messages=[model_request])  # type: ignore
