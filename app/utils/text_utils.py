from __future__ import annotations

import re

import tiktoken


_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    tokens = _TOKENIZER.encode(text)
    if len(tokens) <= max_tokens:
        return text
    truncated = _TOKENIZER.decode(tokens[:max_tokens])
    # Cut at last sentence boundary to avoid mid-sentence truncation
    last_period = truncated.rfind(".")
    if last_period > len(truncated) * 0.7:
        return truncated[: last_period + 1]
    return truncated


def normalize_whitespace(text: str) -> str:
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()
