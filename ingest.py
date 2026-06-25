"""Build the vector store.

Loads a sample of Paul Graham essays, splits them into overlapping word
windows, embeds each chunk with OpenAI, and stores everything in a persistent
ChromaDB collection. Run this once before benchmarking.
"""

import time

import chromadb
from datasets import load_dataset
from openai import OpenAI

from config import settings

NUM_ESSAYS = 50
CHUNK_SIZE = 500  # words per chunk
CHUNK_OVERLAP = 50  # words shared between consecutive chunks
EMBED_BATCH = 100  # chunks written to Chroma per insert

openai = OpenAI(api_key=settings.openai_api_key)


def chunk_text(text: str) -> list[str]:
    """Split text into overlapping windows of words."""
    words = text.split()
    step = CHUNK_SIZE - CHUNK_OVERLAP
    return [
        " ".join(words[start : start + CHUNK_SIZE])
        for start in range(0, len(words), step)
    ]


def embed(text: str) -> list[float]:
    response = openai.embeddings.create(model=settings.embedding_model, input=text)
    return response.data[0].embedding


def main() -> None:
    settings.require("openai_api_key")

    print(f"Loading {NUM_ESSAYS} Paul Graham essays...")
    dataset = load_dataset("sgoel9/paul_graham_essays", split="train")
    essays = dataset.select(range(NUM_ESSAYS))

    print("Resetting ChromaDB collection...")
    chroma = chromadb.PersistentClient(path=settings.chroma_path)
    try:
        chroma.delete_collection(settings.collection_name)
    except Exception:
        pass  # first run — nothing to delete yet
    collection = chroma.create_collection(settings.collection_name)

    print("Chunking and embedding...")
    documents, ids, embeddings = [], [], []
    for essay_idx, essay in enumerate(essays):
        chunks = chunk_text(essay["text"])
        for chunk_idx, chunk in enumerate(chunks):
            documents.append(chunk)
            ids.append(f"essay_{essay_idx}_chunk_{chunk_idx}")
            embeddings.append(embed(chunk))
            time.sleep(0.1)  # stay under the embedding rate limit
        print(f"  essay {essay_idx + 1}/{NUM_ESSAYS} -> {len(chunks)} chunks")

    print(f"\nStoring {len(documents)} chunks...")
    for start in range(0, len(documents), EMBED_BATCH):
        end = start + EMBED_BATCH
        collection.add(
            documents=documents[start:end],
            embeddings=embeddings[start:end],
            ids=ids[start:end],
        )

    print(f"\nDone — {collection.count()} chunks stored in ChromaDB.")


if __name__ == "__main__":
    main()
