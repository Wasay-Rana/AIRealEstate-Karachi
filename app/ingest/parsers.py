from __future__ import annotations

import base64
import tempfile
from pathlib import Path
from typing import Any

import httpx

from app.core.logging import get_logger
from app.utils.text_utils import normalize_whitespace

logger = get_logger(__name__)


class PDFParser:
    async def parse(self, content: str, metadata: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """content is base64-encoded PDF bytes."""
        import pymupdf4llm

        pdf_bytes = base64.b64decode(content)
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name

        try:
            markdown = pymupdf4llm.to_markdown(tmp_path)
            text = normalize_whitespace(markdown)
            meta = {**metadata, "source_type": "pdf", "size_bytes": len(pdf_bytes)}
            logger.info(f"Parsed PDF: {len(text)} chars")
            return text, meta
        finally:
            Path(tmp_path).unlink(missing_ok=True)


class TextParser:
    async def parse(self, content: str, metadata: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        text = normalize_whitespace(content)
        meta = {**metadata, "source_type": "text"}
        return text, meta


class URLParser:
    async def parse(self, content: str, metadata: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        from markdownify import markdownify

        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            response = await client.get(content)
            response.raise_for_status()

        html = response.text
        # Extract title
        title = ""
        if "<title>" in html.lower():
            start = html.lower().index("<title>") + 7
            end = html.lower().index("</title>", start)
            title = html[start:end].strip()

        md = markdownify(html, strip=["script", "style", "nav", "footer", "header"])
        text = normalize_whitespace(md)
        meta = {**metadata, "source_type": "url", "url": content, "title": title}
        logger.info(f"Parsed URL {content}: {len(text)} chars")
        return text, meta


class ParserFactory:
    _parsers = {
        "pdf": PDFParser,
        "text": TextParser,
        "url": URLParser,
    }

    @classmethod
    def get(cls, source_type: str):
        parser_cls = cls._parsers.get(source_type)
        if parser_cls is None:
            raise ValueError(f"Unknown source_type: {source_type}")
        return parser_cls()
