"""MCP registration for the Tool Box challenge."""

from fastmcp import FastMCP

from challenges.tool_box.routing import choose_next_node
from challenges.tool_box.study_retrieval import retrieve_study_passages
from challenges.tool_box.tools import (
    calculate,
    evaluate_expression,
    get_name,
    recognize_shape,
)


mcp = FastMCP("Tool Box")
mcp.tool(
    get_name,
    name="answer_name_question",
    description=(
        "Use when asked for your name or what you are called. Pass the complete "
        "user question verbatim in the question parameter. The returned string "
        "is the complete final answer; answer with it verbatim."
    ),
)
mcp.tool(
    evaluate_expression,
    description=(
        "Use when the original arithmetic question contains two or more "
        "operators. Pass the complete expression exactly once, without solving "
        "or splitting it. Applies standard operator precedence and parentheses."
    ),
)
mcp.tool(
    calculate,
    description=(
        "Use only for a calculation containing exactly one operator, including "
        "an intermediate shape-count calculation. If the original arithmetic "
        "question has multiple operators, use evaluate_expression instead."
    ),
)
mcp.tool(recognize_shape)
mcp.tool(
    retrieve_study_passages,
    description=(
        "Use for School Days factual-recall questions and for finding the STOP_XX "
        "node associated with a named school-trip place. Pass the complete question "
        "once and answer from the returned source passages; the passages are evidence, "
        "not a prewritten final answer."
    ),
)
mcp.tool(
    choose_next_node,
    description=(
        "Use at every step of a School Days journey. Pass the exact map_id, current "
        "node, destination node, and the question's current hops_remaining value (or "
        "null when none is specified). Returns one adjacent next node on the cheapest "
        "directed route, counting edge weights plus the toll of every entered node."
    ),
)


# The parent application mounts this ASGI app at the evaluator's root /mcp.
mcp_app = mcp.http_app(path="/")
