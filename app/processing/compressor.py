from __future__ import annotations

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.internal import RerankedDoc
from app.utils.text_utils import count_tokens, truncate_to_tokens

logger = get_logger(__name__)


class ContextCompressor:
    @staticmethod
    def compress(
        docs: list[RerankedDoc],
        max_tokens: int | None = None,
    ) -> list[RerankedDoc]:
        """Greedy token-budget packing. Truncates the last fitting doc if needed."""
        settings = get_settings()
        budget = max_tokens or settings.max_context_tokens

        result: list[RerankedDoc] = []
        used = 0

        for doc in docs:
            tokens = count_tokens(doc.text)
            if used + tokens <= budget:
                result.append(doc)
                used += tokens
            elif used < budget:
                # Partial fit — truncate last doc
                remaining = budget - used
                truncated_text = truncate_to_tokens(doc.text, remaining)
                trimmed = doc.model_copy(update={"text": truncated_text})
                result.append(trimmed)
                used += count_tokens(truncated_text)
                break
            else:
                break

        logger.debug(f"Compressor: {len(docs)} → {len(result)} docs, {used}/{budget} tokens")
        return result
