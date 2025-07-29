from langchain_core.messages import HumanMessage
from langgraph.types import Command
from langgraph.graph import END, START, StateGraph

from src.utils import State, get_llm

llm = get_llm()


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
