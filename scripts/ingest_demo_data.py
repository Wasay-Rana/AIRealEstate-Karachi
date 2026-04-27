#!/usr/bin/env python3
"""
One-shot script to ingest all demo documents from data/documents/ into the system.
Run after starting the API server: python scripts/ingest_demo_data.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import httpx

BASE_URL = "http://localhost:8000"
DOCS_DIR = Path(__file__).parent.parent / "data" / "documents"


async def ingest_file(client: httpx.AsyncClient, path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    payload = {
        "source_type": "text",
        "content": text,
        "metadata": {"source": path.name, "filename": path.name},
        "namespace": "demo",
        "background": False,  # Wait for completion
    }
    response = await client.post(f"{BASE_URL}/api/v1/ingest", json=payload, timeout=300)
    response.raise_for_status()
    return response.json()


async def main() -> None:
    docs = sorted(DOCS_DIR.glob("*.txt"))
    if not docs:
        print(f"No .txt files found in {DOCS_DIR}")
        sys.exit(1)

    print(f"Ingesting {len(docs)} documents from {DOCS_DIR}\n")

    async with httpx.AsyncClient() as client:
        # Check health first
        try:
            health = await client.get(f"{BASE_URL}/api/v1/health", timeout=10)
            health_data = health.json()
            print(f"Server status: {health_data['status']}")
            print(f"Components: {health_data['components']}\n")
        except Exception as e:
            print(f"Cannot reach server at {BASE_URL}: {e}")
            print("Start the server first: make run")
            sys.exit(1)

        for doc_path in docs:
            print(f"  Ingesting {doc_path.name}...", end=" ", flush=True)
            try:
                result = await ingest_file(client, doc_path)
                status = result.get("status", "unknown")
                chunks = result.get("chunks_created", "?")
                print(f"✓ {status} ({chunks} chunks)")
            except Exception as e:
                print(f"✗ FAILED: {e}")

    print("\nIngestion complete. Run example queries with:")
    print("  python scripts/run_example_queries.py")


if __name__ == "__main__":
    asyncio.run(main())
