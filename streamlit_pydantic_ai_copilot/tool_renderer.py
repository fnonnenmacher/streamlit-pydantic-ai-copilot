from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import logging
import time
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.output import OutputDataT
from typing_extensions import TypeVar
from typing import Any, Generic

from pydantic_ai.tools import ToolDefinition
from pydantic_ai.tools import GenerateToolJsonSchema, _function_schema
import inspect

from streamlit_pydantic_ai_copilot.utils import to_kebab_case

logger = logging.getLogger(__name__)


ToolOuputT = TypeVar("ToolOuputT", default=str)


class ToolStatus(Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ToolResult(Generic[ToolOuputT]):
    status: ToolStatus = ToolStatus.RUNNING
    _output: ToolOuputT | None = None

    def __init__(self, status: ToolStatus = ToolStatus.RUNNING, output: ToolOuputT | None = None):
        self.status = status
        self._output = output

    def wait_for_completion(self) -> ToolStatus:
        while self.status == ToolStatus.RUNNING:
            time.sleep(0.1)
        return self.status

    def complete(self, output: ToolOuputT) -> None:
        self.status = ToolStatus.COMPLETED
        self._output = output

    @property
    def output(self) -> ToolOuputT:
        if self.status == ToolStatus.RUNNING:
            raise ValueError("Tool is still running. Call wait_for_completion() first.")
        if self.status == ToolStatus.FAILED:
            raise ValueError("Tool failed")
        if self._output is None:
            raise ValueError("Tool returned None")
        return self._output


@dataclass
class ToolRenderer(ABC, Generic[ToolOuputT]):
    name: str

    def __init__(self, name: str):
        self.name = name
        self._validate_render_method_signatures()

    def _validate_render_method_signatures(self) -> None:
        """Validate that the render method of child classes have the same method signature as superclass."""

        # Skip validation for the base class itself
        if self.__class__ == StreamlitToolDefinition:
            return

        base_render_sig = inspect.signature(StreamlitToolDefinition.render)
        base_render_params = base_render_sig.parameters.copy()
        base_render_params.pop("self")
        current_render_params = inspect.signature(self.render).parameters

        if len(base_render_params) != len(current_render_params):
            raise TypeError(
                f"Number of paramaters in {self.__class__.__name__}.render do not match the expected signature.\n"
                f"{str(base_render_sig)}\n"
                f"Injecting the paramaters will fail."
            )

    @abstractmethod
    def render(self, args: dict[str, Any], result: ToolResult[ToolOuputT]) -> None:
        raise NotImplementedError


class StreamlitToolDefinition(ToolRenderer[ToolOuputT]):
    name: str
    schema: _function_schema.FunctionSchema

    def __init__(self, name: str | None = None):
        self.name = name or to_kebab_case(self.__class__.__name__)
        self.schema = _function_schema.function_schema(
            self.handle,
            GenerateToolJsonSchema,
            require_parameter_descriptions=True,
            takes_ctx=False,
        )

    @abstractmethod
    def handle(self, *args: Any, **kwargs: Any) -> Any | None:
        raise NotImplementedError

    def as_tool_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name, description=self.schema.description, parameters_json_schema=self.schema.json_schema
        )


class AgentResultRenderer(ABC, Generic[OutputDataT]):
    @abstractmethod
    def render(self, result: AgentRunResult[OutputDataT]) -> None:
        raise NotImplementedError
