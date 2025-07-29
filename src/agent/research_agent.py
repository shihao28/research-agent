import os
from pydantic import BaseModel, Field
from typing import List, Literal

from langchain_community.utilities import SerpAPIWrapper
from langchain_core.tools import StructuredTool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from langgraph.graph import START, StateGraph

from src.utils import State, get_llm, make_supervisor_node

llm = get_llm()

serpapi_tool = SerpAPIWrapper(
    params={
        "engine": "google",
        "google_domain": "google.com",
        "hl": "en",
        "gl": "us",
    },
    serpapi_api_key=os.getenv("SERPAPI_API_KEY"),
)


class SearchResult(BaseModel):
    title: str = Field(..., description="Page title")
    link: str = Field(..., description="Canonical URL")
    snippet: str = Field(..., description="Search snippet")


def _search(query: str, k: int = 5) -> List[SearchResult]:
    raw = serpapi_tool.results(query)
    organic = raw.get("organic_results", [])[:k]
    return [
        SearchResult(
            title=hit["title"], link=hit["link"], snippet=hit.get("snippet", "")
        )
        for hit in organic
    ]


web_search = StructuredTool.from_function(
    name="web_search",
    description="Google via SerpAPI – returns top results with link so the LLM can cite",
    func=_search,
    return_schema=List[SearchResult],
)

SEARCH_SYS_PROMPT = """
You are a careful research assistant.
Whenever you use the `web_search` tool, you **must** cite the fact by
appending the markdown link from the `link` field right after the sentence,
e.g. (https://example.com).
Only cite links that actually support the preceding claim.
"""
search_agent = create_react_agent(llm, tools=[web_search], prompt=SEARCH_SYS_PROMPT)


def search_node(state: State) -> Command[Literal["supervisor"]]:
    result = search_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="search")
            ],
        },
        goto="supervisor",
    )


research_supervisor_node = make_supervisor_node(llm, ["search"])

research_builder = StateGraph(State)
research_builder.add_node("supervisor", research_supervisor_node)
research_builder.add_node("search", search_node)

research_builder.add_edge(START, "supervisor")
research_graph = research_builder.compile()
