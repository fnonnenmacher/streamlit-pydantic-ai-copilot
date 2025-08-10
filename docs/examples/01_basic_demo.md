# Basic Demo

This basic demo demonstrates delegating tool calls to the user and rendering of Pydantic internal tool calls.
It's full code can be found [here](https://github.com/fnonnenmacher/streamlit-pydantic-ai-copilot/examples/01_basic_demo.py).

![type:video](site:resources/01_basic_demo.mp4)

## Run the example

Before running the example, ensure that your OpenAI API key is set in the `OPENAI_API_KEY` environment variable.
Once configured, you can start the demo using the following command:

<!-- termynal -->
```bash
uv run sync --all-groups
uv run streamlit run examples/01_basic_demo.py
```

## The Code

### Defining the agent
First, let's define a [basic Pydantic AI agent](https://ai.pydantic.dev/agents/#running-agents) along with a tool that we want to display in the UI later.

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic import BaseModel
from datetime import datetime
from textwrap import dedent

basic_demo_agent = Agent(
    model=OpenAIModel("gpt-4o"), # (1)
    system_prompt=dedent(
        """\
        You are a helpful assistant. Always call the user by their name.
        Use the ask_user_for_name tool to get the user's name."""
    ),
    output_type=str,
)


class CurrentDateResponse(BaseModel): # (2)
    date: str


@basic_demo_agent.tool_plain()
def get_current_time() -> CurrentDateResponse:
    time.sleep(5) # (3)
    return CurrentDateResponse(date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
```

1.  This example uses an OpenAI model, but you can substitute any other supported model provider as needed.
2.  The wrapper class for the tool result is included for demonstration purposes; in practice, you can return the result directly if preferred.
3.  The call to `sleep` is used to mimic a long-running tool operation.

### Defining a Stremalit tool to collect user input

Next, we define the Streamlit tool that will prompt the user for input.

```python
class AskUserForNameTool(StreamlitToolDefinition[str]): # (1)
    def chat_block(self) -> DeltaGenerator:
        tool_message = st.chat_message("tool", avatar="🔧")
        tool_message.write("Tool Call: _Asking the user for their name_")
        return tool_message

    def handle(self) -> str | None: # (2)
        """
        Asks the user for their name.
        """
        return self.chat_block().text_input("What is your name?")

    def render(self, args: dict[str, Any], result: ToolResult[str]): # (3)
        self.chat_block().write(f"User has entered name: {result.output}")
```

1.  To define a Streamlit tool, create a subclass of `StreamlitToolDefinition`.
    This class is generic, allowing you to specify the output type for your tool.
2.  The `handle` method is responsible for collecting user input.
3.  The `render` method is invoked after the tool has completed.
    Use it to display the user input or any additional information related to the tool's result.

To define a Streamlit tool, create a subclass of `StreamlitToolDefinition`. This class is generic, allowing you to specify the output type for your tool.

You can define any parameters you need—these will be automatically mapped from the tool call.
The docstring (pydoc comment) you provide for the `handle` method will be made available to the agent,
giving it additional context about the tool's purpose and usage.
Inside `handle`, you can use any Streamlit UI elements. Once the method returns a value other than `None`, the tool call is considered complete.
The result is cached, so `handle` will not be called again on subsequent `st.rerun()` events.

Instead, the `render` method is invoked after the tool has completed.
Use it to display the user input or any additional information related to the tool's result.

### Define render logic for Pydantic Internal tools

To define how the output of a Pydantic AI tool is displayed in Streamlit, create a subclass of `ToolRenderer` with the same output type as your Pydantic AI tool. Note that this type consistency is not enforced at compile time, so ensure they match to avoid runtime errors.

```python
class GetCurrentTimeToolRenderer(ToolRenderer[CurrentDateResponse]):
    def __init__(self):
        super().__init__("get_current_time") # (1)

    def render(self, args: dict[str, Any], result: ToolResult[CurrentDateResponse]):
        with st.chat_message("tool", avatar="🔧"):
            st.write(f'_Calling get current time tool with args "{args}"_')
            with st.spinner("Waiting for tool...", show_time=True):
                result.wait_for_completion() # (2)
            assert result.output is not None
            st.write(f"_Tool returned: {result.output.date}_")
```

1. Ensure the tool name needs to matches the Pydantic AI tool name
2. wait for the tool to complete.

It's important to note that the tool name used in your `ToolRenderer` must exactly match the name of the corresponding Pydantic AI tool. If you define your tool using the `@agent.tool` decorator, the tool name will be the method name.

To provide a transparent user experience, this library triggers the `render` method as soon as the tool is called. However, at this point, the tool may not have finished executing, so the result might not be immediately available. Therefore, always call `result.wait_for_completion()` before accessing the result to ensure it is ready. For a better user experience, you can combine this with a Streamlit spinner, as demonstrated in the example above.

## Agent Session

Now let's wire it all together and create a `AgentSession`:

```python
@st.cache_resource # (1)
def init_agent_session() -> AgentSession[None, str]:
    return AgentSession(basic_demo_agent, [AskUserForNameTool(), GetCurrentTimeToolRenderer()])


agent_conversation = init_agent_session().get_or_create_conversation() # (2)
```

1. The `@st.cache_resource` decorator is important because it ensures that the agent session persists across Streamlit reruns, preventing the session from being recreated on every interaction.
2. You can create a new conversation by providing a custom ID

The `@st.cache_resource` decorator is crucial for ensuring that your agent session persists across Streamlit reruns, so the session isn't recreated with every user interaction. When initializing the session, be sure to pass in your previously defined tools and tool renderers as arguments.

The `AgentSession` acts as the main entry point for managing conversations. To start or resume a conversation, call the `get_or_create_conversation()` method. You can optionally provide a custom conversation ID to allow users to have separate, individualized conversations.  By default, `get_or_create_conversation()` will generate a conversation tied to the current Streamlit session, ensuring continuity for each user.

## Streamlit App

Finally, let's put everything together and build the Streamlit app:

```python
st.title("Basic Demo")

for event in agent_conversation.stream(): # (1)
    event.render()

if prompt := st.chat_input("Say something"):
    agent_conversation.run(prompt)
    st.rerun()
```

1. This is a blocking iterator as long as the agent runs

As you can see, the final code is quite straightforward. Like interacting with an agent directly, the process begins with an `st.chat_input`. However, instead of calling the `run` method on the agent itself, you now invoke it on the `agent_conversation` object. The `agent_conversation.stream()` method then yields all ongoing agent events in a blocking fashion, allowing you to render each event with a single command. Behind the scenes, this mechanism automatically calls the previously defined `render` methods for each event, ensuring a seamless and interactive user experience.
