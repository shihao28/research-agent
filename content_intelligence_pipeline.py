import os
from typing import Annotated, List, Literal, Optional

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import StructuredTool, tool
from langchain_experimental.utilities import PythonREPL
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME"),
    temperature=0.3,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    max_tokens=1024,
)


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


########## RESEARCH AGENT ##########
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


########## ANALYSIS AGENT ##########
repl = PythonREPL()


def api_call(topic):
    """This is a dummy function to simulate API call to get statistics"""
    date_range = pd.date_range(end="2025-05-01", periods=10, freq="D")

    # Generate random walk data
    data = np.random.randn(10).cumsum()

    # Create a DataFrame
    df = pd.DataFrame(data, index=date_range, columns=["value"])
    return df.to_dict()


@tool
def get_stats(topic: Annotated[str, "The topic to get statistics for"]):
    """Use this to call the respective API to get the statistics"""
    stats = api_call(topic)
    return stats


@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    return (
        result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."
    )


analysis_generator_agent = create_react_agent(
    llm,
    [get_stats, python_repl_tool],
    prompt=("You should get statistics, generate charts and save it as chart.png."),
)


########## DOC WRITER AGENT ##########
def outline_step(state: State) -> State:
    """Ask the LLM for a bullet‑point outline."""
    outline = llm.invoke(
        [
            (
                "system",
                f"Return ONLY a concise bullet‑point outline for the user query based on the research results",
            ),
            *state["messages"],
        ]
    ).content.strip()

    return Command(
        update={
            "messages": [HumanMessage(content=outline, name="outline_creator")],
            "outline": outline,
        },
        goto="doc_writer",
    )


def draft_step(state: State) -> State:
    """Draft the paper, keeping section order & citations."""
    draft = llm.invoke(
        "Draft the paper following the outline and research results.\n\n"
        f"OUTLINE:\n{state['outline']}\n\n"
        f"RESEARCH RESULTS:\n{state['research_results']}\n\n"
        "Do not change section order or headings. Keep all citations."
    ).content.strip()

    return Command(
        update={"messages": [HumanMessage(content=draft, name="doc_writer")]}, goto=END
    )


# Start with creating outline, then draft report
paper_writing_builder = StateGraph(State)
paper_writing_builder.add_node("outline_creator", outline_step)
paper_writing_builder.add_node("doc_writer", draft_step)
paper_writing_builder.add_edge(START, "outline_creator")
paper_writing_builder.add_edge("outline_creator", "doc_writer")
paper_writing_graph = paper_writing_builder.compile()


########## QUALITY AGENT ##########
@tool
def assess_quality(
    user_request: Annotated[str, "The ORIGINAL user query the draft must fulfil"],
    draft: Annotated[str, "The complete draft produced by writer_agent"],
    outline: Annotated[str, "The bullet‑point outline the draft should follow"],
    research_results: Annotated[str, "The research notes that justify the claims"],
) -> str:
    """
    Return a markdown quality report.

    In addition to structure, factual alignment, citations, style, etc.,
    include a **Query‑Relevance** section that scores how completely the
    draft fulfils the user_request.
    """
    prompt = f"""
You are a senior medical editor.

**Task**  
Compare the DRAFT with the OUTLINE, RESEARCH RESULTS, **and the USER REQUEST**.
Score each category 1‑10, list concrete issues, and finish with a PASS / REVISE
flag.  Important: in *Query‑Relevance* judge whether every element the user
asked for is addressed in adequate depth.

**Output format**

Quality Report
Overall‑score: <0‑10>

Query‑Relevance
…

Structure
…

Factual accuracy
…

Citation check
…

Language & style
…

Recommendation
PASS | REVISE + short rationale
---

### USER REQUEST
{user_request}

---

### OUTLINE
{outline}

---

### RESEARCH RESULTS
{research_results}

---

### DRAFT
{draft}
"""
    return llm.invoke(prompt).content.strip()


QUALITY_SYS_PROMPT = """
You are the quality‑checker.
Call `assess_quality` **exactly once** and return its markdown output verbatim.
"""

quality_checker_agent = create_react_agent(
    llm,
    tools=[assess_quality],
    prompt=QUALITY_SYS_PROMPT,
)

########## FULL GRAPH ##########
coordinator_agent = make_supervisor_node(
    llm, ["research_agent", "analysis_agent", "writing_agent", "quality_agent"]
)


def research_agent(state: State) -> Command[Literal["coordinator_agent"]]:
    response = research_graph.invoke({"messages": state["messages"][-1]})
    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=response["messages"][-1].content, name="research_team"
                )
            ],
            "research_results": response["messages"][-1].content.strip(),
        },
        goto="coordinator_agent",
    )


def analysis_agent(state: State) -> Command[Literal["coordinator_agent"]]:
    """Call the analysis_agent to generate chart and report back to the coordinator."""
    response = analysis_generator_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=response["messages"][-1].content, name="analysis_generation"
                )
            ],
            "analysis_code": response["messages"][-2].content,
        },
        goto="coordinator_agent",
    )


def writing_agent(state: State) -> Command[Literal["coordinator_agent"]]:
    response = paper_writing_graph.invoke(
        {"messages": state["messages"], "research_results": state["research_results"]}
    )
    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=response["messages"][-1].content, name="writing_team"
                )
            ],
            "outline": response["outline"],
            "report": response["messages"][-1].content,
        },
        goto="coordinator_agent",
    )


def quality_agent(state: State) -> Command[Literal["coordinator_agent"]]:
    """Call the quality_checker_agent and report back to the coordinator."""
    response = quality_checker_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=response["messages"][-1].content, name="quality_checker"
                )
            ],
            "quality_check_report": response["messages"][-1].content,
        },
        goto="coordinator_agent",
    )


# Define the graph
super_builder = StateGraph(State)
super_builder.add_node("coordinator_agent", coordinator_agent)
super_builder.add_node("research_agent", research_agent)
super_builder.add_node("analysis_agent", analysis_agent)
super_builder.add_node("writing_agent", writing_agent)
super_builder.add_node("quality_agent", quality_agent)

super_builder.add_edge(START, "coordinator_agent")
super_graph = super_builder.compile()


########## FASTAPI ##########
app = FastAPI(title="Content Intelligence Pipeline API")

class QueryRequest(BaseModel):
    request: str

@app.post("/generate-insight")
def generate_report(payload: QueryRequest):
    try:
        response = super_graph.invoke({
            "messages": [
                ("user", payload.request)
            ]},
            {"recursion_limit": 20}
        )
        return {"report": response['report']}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "POST JSON {'request': 'your prompt'} to /generate-report to retrieve a report."
    }


if __name__ == "__main__":
    uvicorn.run(
        "content_intelligence_pipeline:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=False,
    )
