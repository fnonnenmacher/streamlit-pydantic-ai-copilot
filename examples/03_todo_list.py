import copy
from dataclasses import dataclass
import time
from pydantic_ai import (
    Agent,
    RunContext,
)
from pydantic_ai.settings import ModelSettings

from streamlit_pydantic_ai_copilot import AgentSession
import streamlit as st

from pydantic_ai.models.openai import OpenAIModel

from textwrap import dedent
from dotenv import load_dotenv

from streamlit_pydantic_ai_copilot.agent_conversation import AgentConversation

st.set_page_config(layout="wide")

load_dotenv()


@dataclass
class TodoListItem:
    id: int
    done: bool
    description: str


@dataclass
class TodoList:
    items: list[TodoListItem]


todo_agent: Agent[TodoList, str] = Agent[TodoList, str](
    model=OpenAIModel("gpt-4o"),
    system_prompt=dedent(
        """\
        You are a helpful assitant helping to maintain a todo list.
        """
    ),
    deps_type=TodoList,
    model_settings=ModelSettings(parallel_tool_calls=False),  # we disbale it to visualise the automatic updates
)


@todo_agent.tool
def get_todo_list(ctx: RunContext[TodoList]) -> list[TodoListItem]:
    return ctx.deps.items


@todo_agent.tool()
def add_todo(ctx: RunContext[TodoList], description: str) -> bool:
    time.sleep(2)
    ctx.deps.items.append(TodoListItem(id=len(ctx.deps.items), done=False, description=description))
    return True


def on_agent_deps_change(deps: TodoList):
    st.session_state["todo_list"] = deps
    st.rerun()


@st.cache_resource
def init_agent_session() -> AgentSession[TodoList, str]:
    return AgentSession[TodoList, str](
        agent=todo_agent,
        on_deps_change=on_agent_deps_change,
    )


agent_conversation: AgentConversation[TodoList, str] = init_agent_session().get_or_create_conversation()

st.title("Todo List Agent")

if "todo_list" not in st.session_state:
    st.session_state["todo_list"] = TodoList(items=[])

todo_list: TodoList = st.session_state["todo_list"]

col_todo, col_chat = st.columns([0.6, 0.4])

with col_todo:
    for item in todo_list.items:
        st.checkbox(item.description, value=item.done, key=f"todo_{item.id}")

    with st.form("Add a new todo item", clear_on_submit=True):
        description = st.text_input("Add a new todo item")

        # Every form must have a submit button.
        submitted = st.form_submit_button("Add")
        if submitted:
            new_item = TodoListItem(id=len(todo_list.items), done=False, description=description)
            todo_list.items.append(new_item)
            st.session_state["todo_list"] = todo_list
            st.rerun()

with col_chat:
    chat_container = st.container(border=True, height=500)

    if prompt := st.chat_input("Ask the agent to modify the todo list!"):
        todo_list_copy = copy.deepcopy(todo_list)
        agent_conversation.run(prompt, deps=todo_list_copy)

    with chat_container:
        for event in agent_conversation.stream():
            event.render()
