from __future__ import annotations

import asyncio
from typing import Any, Coroutine


async def gather_with_concurrency(n: int, *coros: Coroutine) -> list[Any]:
    semaphore = asyncio.Semaphore(n)

    async def limited(coro: Coroutine) -> Any:
        async with semaphore:
            return await coro

    return await asyncio.gather(*[limited(c) for c in coros])
