from __future__ import annotations

import os
from typing import Any

import anthropic

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_ENTITY_EXTRACTION_SYSTEM = (
    "You are an expert at extracting structured knowledge from Karachi real estate documents. "
    "Extract entities and their relationships accurately and concisely. "
    "Focus on these entity types: "
    "Society (housing scheme name like DHA City Karachi, Bahria Town Karachi, Malir City), "
    "Sector (numbered sectors in DCK e.g. Sector 14), "
    "Precinct (numbered precincts in BTK e.g. Precinct 19), "
    "Block (lettered blocks in MCDA e.g. Block H), "
    "PlotFile (file numbers and allotment records), "
    "FileStatus (allotted/transferred/possession_given/under_litigation/on_hold/disputed/cancelled), "
    "PossessionRecord (when and where possession was handed over), "
    "DutyRecord (SRB stamp duty, CVT, challan numbers and rates), "
    "PriceIndex (price per sq yard by quarter and source), "
    "LitigationFlag (court cases, case numbers, restrictions on files), "
    "DigitalNOC (SBCA NOC number, type, verification URL), "
    "RegulatoryAuthority (DHA, Bahria Town, MCDA, SRB, SBCA, FBR, SBoR), "
    "TaxRate (stamp duty slabs, CVT rates, withholding tax under Finance Act year), "
    "Developer (project developer entity), "
    "LegalCase (court name, case number, parties, status), "
    "Person (plot owner, allottee, legal heir), "
    "Organization (cantonment board, housing authority, brokerage)."
)


class LightRAGStore:
    """Wraps LightRAG with Claude LLM + OpenAI embeddings."""

    def __init__(self, embedder) -> None:
        self._embedder = embedder
        self._rag = None

    async def initialize(self) -> None:
        from lightrag import LightRAG
        from lightrag.utils import EmbeddingFunc

        settings = get_settings()
        os.makedirs(settings.lightrag_working_dir, exist_ok=True)

        embedding_func = self._embedder.as_lightrag_func()
        llm_func = self._make_claude_llm_func()

        self._rag = LightRAG(
            working_dir=settings.lightrag_working_dir,
            llm_model_func=llm_func,
            embedding_func=embedding_func,
            llm_model_max_async=2,
            embedding_batch_num=8,
            embedding_func_max_async=4,
            addon_params={
                "entity_types": [
                    "Society", "Sector", "Precinct", "Block", "PlotFile",
                    "FileStatus", "PossessionRecord", "DutyRecord", "PriceIndex",
                    "LitigationFlag", "DigitalNOC", "RegulatoryAuthority",
                    "TaxRate", "Developer", "LegalCase", "Person", "Organization",
                ],
            },
        )
        # LightRAG requires explicit storage initialization (v1.3+)
        if hasattr(self._rag, "initialize_storages"):
            await self._rag.initialize_storages()
        elif hasattr(self._rag, "initialize"):
            await self._rag.initialize()
        logger.info(f"LightRAG initialized at {settings.lightrag_working_dir}")

    def _make_claude_llm_func(self):
        settings = get_settings()

        async def claude_llm_func(
            prompt: str,
            system_prompt: str | None = None,
            history_messages: list[dict] | None = None,
            **kwargs: Any,
        ) -> str:
            client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

            sys_content = system_prompt or _ENTITY_EXTRACTION_SYSTEM
            messages = []
            if history_messages:
                messages.extend(history_messages)
            messages.append({"role": "user", "content": prompt})

            response = await client.messages.create(
                model=settings.claude_model,
                max_tokens=2048,
                system=[
                    {
                        "type": "text",
                        "text": sys_content,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=messages,
            )
            return response.content[0].text

        return claude_llm_func

    @property
    def rag(self):
        if self._rag is None:
            raise RuntimeError("LightRAGStore not initialized — call initialize() first")
        return self._rag

    async def insert_documents(self, texts: list[str]) -> None:
        if not texts:
            return
        await self.rag.ainsert(texts)
        logger.info(f"LightRAG: inserted {len(texts)} document(s)")

    async def query(self, query: str, mode: str = "hybrid", top_k: int = 15) -> str:
        from lightrag import QueryParam

        try:
            result = await self.rag.aquery(
                query,
                param=QueryParam(mode=mode, top_k=top_k),
            )
            return result if isinstance(result, str) else str(result)
        except Exception as e:
            logger.error(f"LightRAG query error: {e}")
            return ""

    def get_graph(self):
        """Return the underlying NetworkX graph if available."""
        try:
            storage = self.rag.chunk_entity_relation_graph
            # NetworkXStorage wraps the real graph in ._graph
            if hasattr(storage, "_graph"):
                return storage._graph
            # Fallback: older versions may expose the graph directly
            if hasattr(storage, "nodes"):
                return storage
        except Exception:
            pass
        return None

    def ping(self) -> bool:
        try:
            settings = get_settings()
            return os.path.isdir(settings.lightrag_working_dir)
        except Exception:
            return False
