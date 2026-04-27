from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # LLM
    anthropic_api_key: str = ""
    claude_model: str = "claude-sonnet-4-6"

    # Embeddings
    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    # Pinecone
    pinecone_api_key: str = ""
    pinecone_index_name: str = "graph-rag-agent"
    pinecone_environment: str = "us-east-1"

    # LightRAG
    lightrag_working_dir: str = "./lightrag_storage"

    # Retrieval
    default_retrieval_mode: str = "balanced"
    max_context_tokens: int = 6000
    reranker_top_n: int = 8
    pinecone_top_k: int = 20
    bm25_top_k: int = 15
    lightrag_top_k: int = 15

    # Reranker
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # App
    log_level: str = "INFO"
    environment: str = "development"
    version: str = "1.0.0"


@lru_cache
def get_settings() -> Settings:
    return Settings()
