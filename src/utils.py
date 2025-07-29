import os
from typing import Literal, Optional
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.types import Command
from langgraph.graph import END, MessagesState


def get_llm():
    llm = ChatOpenAI(
        model=os.getenv("MODEL_NAME"),
        temperature=0.3,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        max_tokens=1024,
    )
    return llm
    
class State(MessagesState):
    next: str
    research_results: Optional[str] = None
    analysis_code: Optional[str] = None
    outline: Optional[str] = None
    report: Optional[str] = None
    quality_check_report: Optional[str] = None


def make_supervisor_node(llm: BaseChatModel, members: list[str]) -> str:
    options = ["FINISH"] + members
    system_prompt = (
        "You are a supervisor tasked with managing a conversation between the"
        f" following workers: {members}. Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status. When finished,"
        " respond with FINISH."
    )

    class Router(TypedDict):
        """Worker to route to next. If no workers needed, route to FINISH."""

        next: Literal[*options]

    def supervisor_node(state: State) -> Command[Literal[*members, "__end__"]]:
        """An LLM-based router."""
        messages = [
            {"role": "system", "content": system_prompt},
        ] + state["messages"]
        response = llm.with_structured_output(Router).invoke(messages)
        goto = response["next"]
        if goto == "FINISH":
            goto = END

        return Command(goto=goto, update={"next": goto})

    return supervisor_node
