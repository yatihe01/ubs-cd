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
from challenges.tool_box.working_life import (
    find_group_meeting_time,
    find_minimum_travel_meeting_point,
    find_open_venues,
    plan_group_outing,
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
    find_open_venues,
    description=(
        "Use for every question asking which places or venues are open for eating "
        "on a particular day at a particular hour. Returns every matching name, "
        "already formatted as the complete comma-separated answer."
    ),
)
mcp.tool(
    find_group_meeting_time,
    description=(
        "Use for every question asking when you and named friends can meet. Pass "
        "only the friends (do not add yourself), the full allowed time range, and "
        "the requested duration. This checks friends' schedules plus your accepted "
        "and tentative email events, and returns the exact best window."
    ),
)
mcp.tool(
    find_minimum_travel_meeting_point,
    description=(
        "Use for every question asking where you and named friends should meet to "
        "minimize total Manhattan travel. Pass your [x,y] as your_x and your_y and "
        "only the friends; their positions are looked up automatically."
    ),
)
mcp.tool(
    plan_group_outing,
    description=(
        "MANDATORY for a combined outing question that asks you and friends to meet "
        "during a time range and then eat. Call this one tool instead of separately "
        "guessing a time, point, or venue. It applies calendar priorities, requires "
        "the venue to be open for the full hour after the meeting, and jointly "
        "minimizes everyone's travel plus the one onward trip to the venue."
    ),
)
mcp.tool(
    choose_next_node,
    description=(
        "MANDATORY for every question containing map_id or asking how to travel from "
        "one node to another. Call immediately; never answer a map route without this "
        "tool. On the first call, copy map_id exactly, use the stated start as "
        "current_node, the requested endpoint as destination, and null for "
        "hops_remaining unless the question supplies a remaining-hop value. The result "
        "is one adjacent node: call again from that node and repeat until destination."
    ),
)
mcp.tool(
    retrieve_study_passages,
    description=(
        "Use for factual recall and for finding the STOP_XX of a named school-trip "
        "place. Pass the complete question and a concise semantic_context rewrite with "
        "likely corpus terms and absent synonyms. Passages are ranked most relevant "
        "first; answer from their evidence."
    ),
)


# The parent application mounts this ASGI app at the evaluator's root /mcp.
mcp_app = mcp.http_app(path="/")
