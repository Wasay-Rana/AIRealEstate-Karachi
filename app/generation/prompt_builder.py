from __future__ import annotations

from app.models.internal import RerankedDoc

_SYSTEM_INSTRUCTION = (
    "You are a precise research assistant answering questions based strictly on the provided context.\n"
    "Rules:\n"
    "1. Answer using ONLY the information in the provided context.\n"
    "2. Cite sources using [Source N] inline — e.g. 'Transformers use self-attention [Source 1].'\n"
    "3. If multiple sources support a point, cite all of them: [Source 1][Source 3].\n"
    "4. If the context is insufficient to answer, say: 'The provided context does not contain enough information to answer this question.'\n"
    "5. Never hallucinate or add information beyond the context.\n"
    "6. Be concise but complete."
)


def build_messages(
    query: str,
    docs: list[RerankedDoc],
) -> tuple[list[dict], list[dict]]:
    """
    Returns (system_blocks, user_messages) in Anthropic API format.
    The static instruction block carries cache_control for prompt caching.
    """
    context_parts = []
    for i, doc in enumerate(docs, start=1):
        source_label = f"[Source {i}] (from: {doc.source})"
        context_parts.append(f"{source_label}\n{doc.text}")

    context_text = "\n\n---\n\n".join(context_parts)

    system_blocks = [
        {
            "type": "text",
            "text": _SYSTEM_INSTRUCTION,
            "cache_control": {"type": "ephemeral"},  # Cache the static instructions
        },
        {
            "type": "text",
            "text": f"Context documents:\n\n{context_text}",
            # Dynamic per-request — not cached
        },
    ]

    messages = [{"role": "user", "content": f"Question: {query}"}]

    return system_blocks, messages
