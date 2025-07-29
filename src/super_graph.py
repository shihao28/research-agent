from typing import Literal

from langchain_core.messages import HumanMessage
from langgraph.graph import START, StateGraph
from langgraph.types import Command

from src.agent.research_agent import research_graph
from src.agent.analysis_agent import analysis_generator_agent
from src.agent.writing_agent import paper_writing_graph
from src.agent.quality_agent import quality_checker_agent
from src.utils import State, make_supervisor_node, get_llm

llm = get_llm()


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
