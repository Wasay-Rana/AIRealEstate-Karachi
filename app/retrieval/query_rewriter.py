from __future__ import annotations

import anthropic

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = (
    "You are a search query optimizer for a technical knowledge base. "
    "Expand and clarify the user's query to maximize retrieval coverage. "
    "Add relevant synonyms and related technical terms. "
    "Return ONLY the rewritten query, nothing else — no explanation, no quotes."
)


class QueryRewriter:
    def __init__(self) -> None:
        settings = get_settings()
        self._client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        self._model = settings.claude_model

    async def rewrite(self, query: str) -> str:
        try:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=200,
                system=[
                    {
                        "type": "text",
                        "text": _SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": query}],
            )
            rewritten = response.content[0].text.strip()
            if rewritten and rewritten != query:
                logger.debug(f"Query rewritten: {query!r} → {rewritten!r}")
                return rewritten
        except Exception as exc:
            logger.warning(f"Query rewrite failed, using original: {exc}")
        return query
