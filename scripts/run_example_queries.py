#!/usr/bin/env python3
"""
Run the 3 example queries from data/example_queries.json and print annotated results.
Run after ingesting demo data: python scripts/run_example_queries.py
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import httpx

BASE_URL = "http://localhost:8000"
QUERIES_FILE = Path(__file__).parent.parent / "data" / "example_queries.json"

_SEPARATOR = "=" * 80


async def run_query(client: httpx.AsyncClient, query_data: dict) -> dict:
    payload = {
        "query": query_data["query"],
        "mode": "auto",
        "namespace": "demo",
        "rewrite_query": True,
    }
    response = await client.post(
        f"{BASE_URL}/api/v1/query",
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def print_result(query_data: dict, result: dict) -> None:
    print(_SEPARATOR)
    print(f"QUERY {query_data['id'].upper()}: {query_data['query']}")
    print(f"Expected mode: {query_data['expected_mode']} | Actual mode: {result['mode_used']}")
    print(f"Latency: {result['latency_ms']}ms")
    if result.get("query_rewritten"):
        print(f"Rewritten: {result['query_rewritten']}")
    print(f"Retrieval: {result['retrieval_breakdown']}")
    print()
    print("ANSWER:")
    print(result["answer"])
    print()

    if result.get("citations"):
        print(f"CITATIONS ({len(result['citations'])}):")
        for i, cit in enumerate(result["citations"], 1):
            print(f"  [{i}] {cit['source']} (score={cit['score']:.4f})")
            print(f"      {cit['text_snippet'][:120]}...")
    print()


async def main() -> None:
    queries = json.loads(QUERIES_FILE.read_text())
    print(f"Running {len(queries)} example queries against {BASE_URL}\n")

    async with httpx.AsyncClient() as client:
        # Health check
        try:
            health = (await client.get(f"{BASE_URL}/api/v1/health", timeout=10)).json()
            print(f"Server: {health['status']} | v{health.get('version', '?')}\n")
        except Exception as e:
            print(f"Server unreachable: {e}\nStart with: make run")
            sys.exit(1)

        for qdata in queries:
            print(f"Running {qdata['id']}...", flush=True)
            try:
                result = await run_query(client, qdata)
                print_result(qdata, result)
            except Exception as e:
                print(f"  FAILED: {e}\n")

    print(_SEPARATOR)
    print("Done. Check answers above for multi-hop reasoning quality.")


if __name__ == "__main__":
    asyncio.run(main())
