"""Smoke test for the external services.

Run this first to confirm every API key works before ingesting or
benchmarking. Each check is independent — a failure in one doesn't stop
the rest.
"""

import chromadb
import cohere
from groq import Groq
from openai import OpenAI

from config import settings


def check_openai() -> str:
    client = OpenAI(api_key=settings.openai_api_key)
    response = client.embeddings.create(
        model=settings.embedding_model, input="This is a test sentence."
    )
    return f"got a {len(response.data[0].embedding)}-dim embedding"


def check_groq() -> str:
    client = Groq(api_key=settings.groq_api_key)
    response = client.chat.completions.create(
        model=settings.generation_model,
        messages=[{"role": "user", "content": "Say hello in one word."}],
    )
    return f"Groq replied: {response.choices[0].message.content.strip()}"


def check_cohere() -> str:
    client = cohere.ClientV2(api_key=settings.cohere_api_key)
    response = client.rerank(
        model=settings.rerank_model,
        query="What is the best way to start a startup?",
        documents=[
            "Focus on solving a real problem people have.",
            "The weather today is sunny and warm.",
            "Startups succeed by finding product market fit early.",
        ],
    )
    top = response.results[0]
    return f"top doc index {top.index}, score {round(top.relevance_score, 3)}"


def check_chromadb() -> str:
    client = chromadb.Client()  # in-memory, just to prove it works
    collection = client.create_collection("smoke_test")
    collection.add(documents=["This is a test document"], ids=["doc1"])
    results = collection.query(query_texts=["test"], n_results=1)
    return f"query returned: {results['documents'][0][0]}"


CHECKS = {
    "OpenAI embeddings": check_openai,
    "Groq chat": check_groq,
    "Cohere rerank": check_cohere,
    "ChromaDB": check_chromadb,
}


def main() -> None:
    print("=== Testing all APIs ===\n")
    for name, check in CHECKS.items():
        try:
            print(f"[ok]   {name} — {check()}")
        except Exception as exc:
            print(f"[fail] {name} — {exc}")
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
