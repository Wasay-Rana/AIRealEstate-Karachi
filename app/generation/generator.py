from __future__ import annotations

from typing import AsyncGenerator

import anthropic

from app.core.config import get_settings
from app.core.exceptions import GenerationError
from app.core.logging import get_logger
from app.generation.citation_formatter import extract_citations
from app.generation.prompt_builder import build_messages
from app.models.internal import RerankedDoc
from app.models.responses import Citation

logger = get_logger(__name__)


class AnswerGenerator:
    def __init__(self) -> None:
        settings = get_settings()
        self._client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        self._model = settings.claude_model

    async def generate(
        self,
        query: str,
        docs: list[RerankedDoc],
    ) -> tuple[str, list[Citation]]:
        if not docs:
            return (
                "No relevant context was found to answer this question.",
                [],
            )

        system_blocks, messages = build_messages(query, docs)

        try:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=2048,
                system=system_blocks,
                messages=messages,
            )
        except anthropic.APIError as exc:
            logger.error(f"Claude API error: {exc}")
            raise GenerationError(f"LLM generation failed: {exc}") from exc

        answer = response.content[0].text
        citations = extract_citations(answer, docs)

        logger.debug(
            f"Generated answer: {len(answer)} chars, {len(citations)} citations | "
            f"cache_read={response.usage.cache_read_input_tokens} "
            f"cache_write={response.usage.cache_creation_input_tokens}"
        )

        return answer, citations

    async def generate_stream(
        self,
        query: str,
        docs: list[RerankedDoc],
    ) -> AsyncGenerator[str, None]:
        if not docs:
            yield "No relevant context was found to answer this question."
            return

        system_blocks, messages = build_messages(query, docs)

        try:
            async with self._client.messages.stream(
                model=self._model,
                max_tokens=2048,
                system=system_blocks,
                messages=messages,
            ) as stream:
                async for text in stream.text_stream:
                    yield text
        except anthropic.APIError as exc:
            logger.error(f"Claude streaming error: {exc}")
            raise GenerationError(f"LLM streaming failed: {exc}") from exc
