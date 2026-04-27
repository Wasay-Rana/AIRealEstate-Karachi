from __future__ import annotations

import hashlib
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.models.internal import Chunk
from app.utils.text_utils import count_tokens


class SemanticChunker:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=count_tokens,
            is_separator_regex=False,
        )

    def chunk(
        self,
        text: str,
        document_id: str,
        metadata: dict[str, Any],
    ) -> list[Chunk]:
        pieces = self._splitter.split_text(text)
        chunks: list[Chunk] = []
        for piece in pieces:
            if not piece.strip():
                continue
            chunk_id = hashlib.sha256(piece.encode()).hexdigest()[:16]
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    text=piece,
                    token_count=count_tokens(piece),
                    metadata={**metadata, "document_id": document_id},
                )
            )
        return chunks
