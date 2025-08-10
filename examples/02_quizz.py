from typing import Any
from pydantic import BaseModel, Field
from pydantic_ai import (
    Agent,
)
from pydantic_ai.agent import AgentRunResult
from streamlit.delta_generator import DeltaGenerator

from streamlit_pydantic_ai_copilot import AgentSession
import streamlit as st
from streamlit_pydantic_ai_copilot import (
    ToolResult,
    StreamlitToolDefinition,
)

from pydantic_ai.models.openai import OpenAIModel

from textwrap import dedent
from dotenv import load_dotenv

from streamlit_pydantic_ai_copilot import AgentResultRenderer
from streamlit_pydantic_ai_copilot.agent_conversation import AgentConversation

load_dotenv()


class QuizzResponse(BaseModel):
    user_message: str = Field(description="The message displayed to the user, keep it engaging and fun")
    number: int = Field(description="The total number of all questions asked so far")
    correct: int = Field(description="The total number of all correct answers given by the user")


quizz_agent: Agent[None, QuizzResponse] = Agent[None, QuizzResponse](
    model=OpenAIModel("gpt-4o"),
    system_prompt=dedent(
        """\
        You are a Quiz Master.
        The user gives you a topic and you will ask one question about that topic with 4 answer options.
        Use the ask-question tool to ask the question and provide possible answer options with only one being right.
        Afterwards, determine if the user has answered the question correctly and give the user feedback.
        Inlcude the total number of questions asked during the complete conversation in your final response.
        """
    ),
    output_type=QuizzResponse,
)


class AskQuestion(StreamlitToolDefinition[str]):
    def chat_block(self) -> DeltaGenerator:
        return st.chat_message("question", avatar="❓")

    def handle(self, question: str, answer_options: list[str]) -> str | None:
        """
        Ask the user a multiple choice question and let him select one of the answer options.

        Args:
            question: The question to ask the user
            answer_options: A list of answer options. Only one of the options is correct.

        Returns:
            The user's answer
        """
        with self.chat_block():
            st.write(question)
            selected_answer = st.radio(f"Question: {question}", answer_options, label_visibility="hidden")
            if st.button("Done", key=f"done_button_{question}"):
                return selected_answer
            return None

    def render(self, args: dict[str, Any], result: ToolResult[str]):
        with self.chat_block():
            st.write(args["question"])
            st.radio(
                label=f"Question: {args['question']}",
                options=args["answer_options"],
                label_visibility="hidden",
                index=args["answer_options"].index(result.output),
                disabled=True,
            )


class QuizzResultRenderer(AgentResultRenderer[QuizzResponse]):
    def render(self, result: AgentRunResult[QuizzResponse]):
        with st.chat_message("assistant"):
            st.write(result.output.user_message)
            st.write(f"_You have answered {result.output.correct} questions correctly out of {result.output.number}._")


@st.cache_resource
def init_agent_session() -> AgentSession[None, QuizzResponse]:
    return AgentSession[None, QuizzResponse](
        agent=quizz_agent,
        tool_renderers=[AskQuestion()],
        result_renderer=QuizzResultRenderer(),
    )


agent_conversation: AgentConversation[None, QuizzResponse] = init_agent_session().get_or_create_conversation()

st.title("Quizz Agent")

for event in agent_conversation.stream():
    event.render()

if prompt := st.chat_input("Tell the agent a topic you want to be quizzed on!"):
    agent_conversation.run(prompt)
    st.rerun()
