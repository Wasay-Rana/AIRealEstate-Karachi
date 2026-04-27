from __future__ import annotations

import re
from typing import Literal

from app.core.logging import get_logger

logger = get_logger(__name__)

_DEEP_PATTERNS = re.compile(
    r"\b(how does .+ relate|compare|difference between|why does|trace|chain|"
    r"multi.hop|step.by.step|explain the relationship|what connects|"
    r"how are .+ connected|impact of .+ on|evolution of|history of)\b",
    re.IGNORECASE,
)

_FAST_MAX_TOKENS = 6


def detect_mode(query: str) -> Literal["fast", "balanced", "deep"]:
    tokens = query.split()

    if len(tokens) <= _FAST_MAX_TOKENS and "?" not in query:
        logger.debug(f"Mode=fast (short query): {query!r}")
        return "fast"

    if _DEEP_PATTERNS.search(query):
        logger.debug(f"Mode=deep (complex pattern): {query!r}")
        return "deep"

    logger.debug(f"Mode=balanced (default): {query!r}")
    return "balanced"
