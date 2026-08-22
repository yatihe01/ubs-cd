"""MCP registration for Stage 1 of the Tool Box challenge."""

from fastmcp import FastMCP

from challenges.tool_box.tools import calculate, get_name, recognize_shape


mcp = FastMCP("Tool Box — Stage 1")
mcp.tool(get_name)
mcp.tool(calculate)
mcp.tool(recognize_shape)


# Mounting this ASGI app at /tool-box/mcp exposes the MCP endpoint there.
mcp_app = mcp.http_app(path="/")
