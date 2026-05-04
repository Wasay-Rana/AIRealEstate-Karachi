from __future__ import annotations

import networkx as nx
from fastapi import APIRouter, Query

from app.core.dependencies import get_lightrag_store
from app.core.logging import get_logger
from app.models.responses import GraphEdge, GraphNode, GraphResponse

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1", tags=["graph"])


@router.get("/graph", response_model=GraphResponse)
async def explore_graph(
    entity: str | None = Query(default=None, description="Filter by entity name (case-insensitive)"),
    depth: int = Query(default=2, ge=1, le=5, description="Hop depth from the entity"),
) -> GraphResponse:
    store = get_lightrag_store()
    graph = store.get_graph()

    if graph is None:
        return GraphResponse(nodes=[], edges=[], total_nodes=0, total_edges=0)

    try:
        if entity:
            # Find matching nodes (case-insensitive prefix match)
            matches = [n for n in graph.nodes if entity.lower() in str(n).lower()]
            if not matches:
                return GraphResponse(nodes=[], edges=[], total_nodes=0, total_edges=0)

            # BFS up to `depth` hops from all matched seed nodes
            subgraph_nodes: set = set()
            for seed in matches:
                subgraph_nodes.update(nx.ego_graph(graph, seed, radius=depth).nodes)
            subgraph = graph.subgraph(subgraph_nodes)
        else:
            # Return the full graph (cap at 200 nodes for response size)
            if len(graph.nodes) > 200:
                nodes_to_keep = list(graph.nodes)[:200]
                subgraph = graph.subgraph(nodes_to_keep)
            else:
                subgraph = graph

        nodes = [
            GraphNode(
                id=str(n),
                label=str(n),
                type=subgraph.nodes[n].get("type", "entity"),
                description=subgraph.nodes[n].get("description", ""),
            )
            for n in subgraph.nodes
        ]

        edges = [
            GraphEdge(
                source=str(u),
                target=str(v),
                label=subgraph.edges[u, v].get("relation", ""),
                weight=float(subgraph.edges[u, v].get("weight", 1.0)),
            )
            for u, v in subgraph.edges
        ]

        return GraphResponse(
            nodes=nodes,
            edges=edges,
            total_nodes=len(nodes),
            total_edges=len(edges),
        )

    except Exception as exc:
        logger.error(f"Graph exploration error: {exc}", exc_info=True)
        return GraphResponse(nodes=[], edges=[], total_nodes=0, total_edges=0)
