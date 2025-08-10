# Streamlit ❤️ Pydantic AI Copilot

Building complex agent applications with Streamlit and Pydantic AI can be challenging, especially when dealing with their different threading models. This Python library simplifies the process by providing a a seamless bridge between Streamlit's synchronous nature and Pydantic AI's asynchronous capabilities. Inspired by the [ag ui protocol](https://docs.ag-ui.com/introduction) and the [copilotKit](https://www.copilotkit.ai/) TypeScript client, this library cuts down on boilerplate code and dramatically accelerates rapid prototyping. It allows you to build powerful, in-process agent workflows without the complexities of a distributed system.

> [!NOTE]
> While inspired by the [ag ui protocol](https://docs.ag-ui.com/introduction) and the [copilotKit](https://www.copilotkit.ai/) TypeScript client, this library is purpose-built for the in-process execution model of Streamlit. For this reason, it is not an implementation of the ag ui protocol.

## Features
* **Delegating Tool Calls**: A structured mechanism for routing agent tool calls to Streamlit UI components for user interaction.
* **Visualizing Internal Tools**: Display Pydantic AI's internal tool calls in the Streamlit frontend for debugging and transparency.
* **Background Agent Execution**: Execute Pydantic AI agents asynchronously, preventing UI blocking and ensuring a smooth user experience.
* **Simplified State Synchronization: Streamlined approach to managing and synchronizing shared state between the Streamlit application and Pydantic AI agents**.
* **Built-in Conversation Context**: Core features for managing conversation history and user sessions, providing persistent context for agent interactions.

## Getting Started

🚧 Under Construction 🚧
