"""Central configuration for the RAG benchmark.

All tunable knobs (models, paths, retrieval depth) live here so the rest of
the codebase reads as plain logic rather than a pile of magic strings.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    # API credentials
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    cohere_api_key: str = os.getenv("COHERE_API_KEY", "")

    # Models
    embedding_model: str = "text-embedding-3-small"
    generation_model: str = "llama-3.1-8b-instant"
    rerank_model: str = "rerank-english-v3.0"
    eval_model: str = "gpt-4o-mini"

    # Vector store
    chroma_path: str = "./chroma_db"
    collection_name: str = "paul_graham"

    # Retrieval
    retrieve_k: int = 40  # candidates pulled from the vector store
    top_n: int = 3  # chunks passed to the LLM after (optional) reranking

    def require(self, *names: str) -> None:
        """Fail fast with a clear message if a required key is missing."""
        missing = [n for n in names if not getattr(self, n)]
        if missing:
            raise RuntimeError(
                f"Missing required environment variable(s): "
                f"{', '.join(n.upper() for n in missing)}. "
                f"Copy .env.example to .env and fill them in."
            )


settings = Settings()
