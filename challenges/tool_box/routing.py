"""Shortest-path next-hop selection for Tool Box Phase 2 journeys."""

from __future__ import annotations

import heapq
import math
from functools import lru_cache
from typing import Any

import httpx


GRAPH_URL = "https://tool-box-2591eaa24fa3.herokuapp.com/graph"


def choose_next_node(
    map_id: str,
    current_node: str,
    destination: str,
    hops_remaining: int | None = None,
) -> str:
    """Return the next node on the cheapest valid route to the destination."""

    if not isinstance(map_id, str) or not map_id.strip():
        raise ValueError("map_id must be a non-empty string")
    adjacency, tolls = _fetch_graph(map_id)
    path = find_cheapest_path(adjacency, tolls, current_node, destination, hops_remaining)
    return path[1]


@lru_cache(maxsize=128)
def _fetch_graph(map_id: str) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    response = httpx.get(
        GRAPH_URL,
        params={"map_id": map_id},
        headers={"Accept": "application/json"},
        timeout=httpx.Timeout(6.0, connect=3.0),
        follow_redirects=True,
    )
    response.raise_for_status()
    return _validate_graph(response.json())


def _validate_graph(payload: Any) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    if not isinstance(payload, dict):
        raise ValueError("graph response must be an object")
    raw_adjacency = payload.get("adjacency")
    raw_tolls = payload.get("tolls")
    if not isinstance(raw_adjacency, dict) or not isinstance(raw_tolls, dict):
        raise ValueError("graph response must contain adjacency and tolls objects")

    adjacency: dict[str, dict[str, float]] = {}
    for node, raw_edges in raw_adjacency.items():
        if not isinstance(node, str) or not isinstance(raw_edges, dict):
            raise ValueError("graph adjacency is invalid")
        adjacency[node] = {}
        for neighbor, raw_weight in raw_edges.items():
            weight = _valid_cost(raw_weight, "edge weight")
            adjacency[node][str(neighbor)] = weight

    tolls = {str(node): _valid_cost(value, "node toll") for node, value in raw_tolls.items()}
    all_nodes = set(adjacency) | {neighbor for edges in adjacency.values() for neighbor in edges}
    if not all_nodes.issubset(tolls):
        raise ValueError("every graph node must have a toll")
    for node in all_nodes:
        adjacency.setdefault(node, {})
    return adjacency, tolls


def _valid_cost(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{label} must be finite and non-negative")
    return result


def find_cheapest_path(
    adjacency: dict[str, dict[str, float]],
    tolls: dict[str, float],
    current_node: str,
    destination: str,
    hops_remaining: int | None = None,
) -> list[str]:
    """Find a directed simple path, charging each entered node's toll."""

    if current_node not in adjacency:
        raise ValueError(f"unknown current node: {current_node}")
    if destination not in adjacency:
        raise ValueError(f"unknown destination node: {destination}")
    if current_node == destination:
        raise ValueError("already at the destination; no next node is needed")
    if hops_remaining is not None:
        if isinstance(hops_remaining, bool) or not isinstance(hops_remaining, int):
            raise ValueError("hops_remaining must be an integer or null")
        if hops_remaining < 1:
            raise ValueError("hops_remaining must allow at least one edge")

    maximum_hops = len(adjacency) - 1
    if hops_remaining is not None:
        maximum_hops = min(maximum_hops, hops_remaining)

    queue: list[tuple[float, int, tuple[str, ...]]] = [(0.0, 0, (current_node,))]
    best: dict[tuple[str, int], float] = {(current_node, 0): 0.0}
    while queue:
        cost, hops, path = heapq.heappop(queue)
        node = path[-1]
        if cost > best.get((node, hops), math.inf):
            continue
        if node == destination:
            return list(path)
        if hops >= maximum_hops:
            continue

        for neighbor, edge_weight in sorted(adjacency[node].items()):
            if neighbor in path:
                continue
            next_hops = hops + 1
            next_cost = cost + edge_weight + tolls[neighbor]
            state = (neighbor, next_hops)
            if next_cost < best.get(state, math.inf):
                best[state] = next_cost
                heapq.heappush(queue, (next_cost, next_hops, path + (neighbor,)))

    allowance = f" within {maximum_hops} hops" if hops_remaining is not None else ""
    raise ValueError(f"no route from {current_node} to {destination}{allowance}")
