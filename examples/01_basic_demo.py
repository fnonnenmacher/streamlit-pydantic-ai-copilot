from typing import Any
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

from streamlit.delta_generator import DeltaGenerator

from streamlit_pydantic_ai_copilot.agent_session import AgentSession
import streamlit as st
from streamlit_pydantic_ai_copilot.agent_events import (
    ToolResult,
    ToolRenderer,
    StreamlitToolDefinition,
)

from pydantic import BaseModel

from datetime import datetime
from textwrap import dedent
import time
from dotenv import load_dotenv

load_dotenv()


basic_demo_agent = Agent(
    model=OpenAIModel("gpt-4o"),
    system_prompt=dedent(
        """\
        You are a helpful assistant. Always call the user by their name.
        Use the ask_user_for_name tool to get the user's name."""
    ),
    output_type=str,
)


class CurrentDateResponse(BaseModel):
    date: str


@basic_demo_agent.tool_plain()
def get_current_time() -> CurrentDateResponse:
    # sleep for 5 seconds to simulate a long running tool call
    time.sleep(5)
    return CurrentDateResponse(date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


class AskUserForNameTool(StreamlitToolDefinition[str]):
    def chat_block(self) -> DeltaGenerator:
        tool_message = st.chat_message("tool", avatar="🔧")
        tool_message.write("Tool Call: _Asking the user for their name_")
        return tool_message

    def handle(self) -> str | None:
        """
        Asks the user for their name.
        """
        return self.chat_block().text_input("What is your name?")

    def render(self, args: dict[str, Any], result: ToolResult[str]):
        self.chat_block().write(f"User has entered name: {result.output}")


class GetCurrentTimeToolRenderer(ToolRenderer[CurrentDateResponse]):
    def __init__(self):
        super().__init__("get_current_time")

    def render(self, args: dict[str, Any], result: ToolResult[CurrentDateResponse]):
        with st.chat_message("tool", avatar="🔧"):
            st.write(f'_Calling get current time tool with args "{args}"_')
            with st.spinner("Waiting for tool...", show_time=True):
                result.wait_for_completion()
            assert result.output is not None
            st.write(f"_Tool returned: {result.output.date}_")


@st.cache_resource
def init_agent_session() -> AgentSession[None, str]:
    return AgentSession(basic_demo_agent, [AskUserForNameTool(), GetCurrentTimeToolRenderer()])


agent_conversation = init_agent_session().get_or_create_conversation()

st.title("Basic Demo")

for event in agent_conversation.stream():
    event.render()

if prompt := st.chat_input("Say something"):
    agent_conversation.run(prompt)
    st.rerun()
