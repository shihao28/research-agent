import pandas as pd
import numpy as np
from typing import Annotated

from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from langgraph.prebuilt import create_react_agent

from src.utils import get_llm

llm = get_llm()


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
