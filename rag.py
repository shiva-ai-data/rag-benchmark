"""Core RAG pipeline.

A single ``RAGPipeline`` implements both variants we benchmark:

* ``rerank=False`` -> baseline: take the top vector-search hits as-is.
* ``rerank=True``  -> improved: rerank the candidates with Cohere, then keep
  the best few.

Keeping both in one class means the retrieval and generation code is written
once and the only thing that changes between experiments is a single flag.
"""

from dataclasses import dataclass

import chromadb
import cohere
from groq import Groq
from openai import OpenAI

from config import settings


@dataclass
class RAGResult:
    question: str
    answer: str
    contexts: list[str]
    scores: list[float] | None = None  # rerank relevance scores, when available


class RAGPipeline:
    def __init__(self, rerank: bool = True):
        self.rerank_enabled = rerank

        settings.require("openai_api_key", "groq_api_key")
        if rerank:
            settings.require("cohere_api_key")

        self.openai = OpenAI(api_key=settings.openai_api_key)
        # Free-tier Groq has a tight tokens-per-minute limit; let the SDK
        # back off and retry on 429s instead of crashing the whole run.
        self.groq = Groq(api_key=settings.groq_api_key, max_retries=8)
        self.cohere = (
            cohere.ClientV2(api_key=settings.cohere_api_key) if rerank else None
        )

        chroma = chromadb.PersistentClient(path=settings.chroma_path)
        self.collection = chroma.get_collection(settings.collection_name)

    def embed(self, text: str) -> list[float]:
        response = self.openai.embeddings.create(
            model=settings.embedding_model,
            input=text,
        )
        return response.data[0].embedding

    def retrieve(self, query: str) -> list[str]:
        """Return the top-k candidate chunks from the vector store."""
        results = self.collection.query(
            query_embeddings=[self.embed(query)],
            n_results=settings.retrieve_k,
        )
        return results["documents"][0]

    def rerank(self, query: str, chunks: list[str]) -> tuple[list[str], list[float]]:
        """Reorder candidates by relevance and keep the best ``top_n``."""
        response = self.cohere.rerank(
            model=settings.rerank_model,
            query=query,
            documents=chunks,
            top_n=settings.top_n,
        )
        ranked = [chunks[r.index] for r in response.results]
        scores = [round(r.relevance_score, 3) for r in response.results]
        return ranked, scores

    def generate(self, query: str, contexts: list[str]) -> str:
        context_block = "\n\n".join(contexts)
        prompt = (
            "Answer the question based only on the context below.\n"
            'If the answer is not in the context, say "I don\'t know."\n\n'
            f"Context:\n{context_block}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )
        response = self.groq.chat.completions.create(
            model=settings.generation_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,  # deterministic generation for reproducible benchmarks
        )
        return response.choices[0].message.content

    def answer(self, query: str) -> RAGResult:
        """Run the full pipeline for a single question."""
        candidates = self.retrieve(query)

        if self.rerank_enabled:
            contexts, scores = self.rerank(query, candidates)
        else:
            contexts, scores = candidates[: settings.top_n], None

        answer = self.generate(query, contexts)
        return RAGResult(query, answer, contexts, scores)
