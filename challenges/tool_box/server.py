"""MCP registration for Stage 1 of the Tool Box challenge."""

from fastmcp import FastMCP

from challenges.tool_box.tools import (
    calculate,
    evaluate_expression,
    get_name,
    recognize_shape,
)


mcp = FastMCP("Tool Box — Stage 1")
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


# The parent application mounts this ASGI app at the evaluator's root /mcp.
mcp_app = mcp.http_app(path="/")
