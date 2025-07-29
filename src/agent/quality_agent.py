from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from src.utils import get_llm

llm = get_llm()

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
