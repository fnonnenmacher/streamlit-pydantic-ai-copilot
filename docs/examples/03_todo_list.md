# Todo List Agent

This example demonstrates how to maintain a shared state between the UI and the agent, allowing the frontend to automatically update whenever the agent modifies the state.
It's full code can be found [here](https://github.com/fnonnenmacher/streamlit-pydantic-ai-copilot/examples/03_todo_list.py).

![type:video](site:resources/03_todo_list.mp4)

## Run the example

Before running the example, ensure that your OpenAI API key is set in the `OPENAI_API_KEY` environment variable.
Once configured, you can start the demo using the following command:

<!-- termynal -->
```bash
uv run sync --all-groups
uv run streamlit run examples/03_todo_list.py
```
