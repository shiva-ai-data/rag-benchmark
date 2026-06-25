import os
import time
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from datasets import load_dataset

load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text):
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def main():
    print("Loading Paul Graham essays...")
    dataset = load_dataset("sgoel9/paul_graham_essays", split="train")
    essays = dataset.select(range(50))
    print(f"Loaded {len(essays)} essays")

    print("Setting up ChromaDB...")
    chroma_client = chromadb.PersistentClient(path="./chroma_db")

    try:
        chroma_client.delete_collection("paul_graham")
    except:
        pass

    collection = chroma_client.create_collection("paul_graham")

    print("Chunking and embedding essays...")
    all_chunks = []
    all_ids = []
    all_embeddings = []

    for i, essay in enumerate(essays):
        text = essay["text"]
        chunks = chunk_text(text)

        for j, chunk in enumerate(chunks):
            chunk_id = f"essay_{i}_chunk_{j}"
            embedding = get_embedding(chunk)
            time.sleep(0.1)
            all_chunks.append(chunk)
            all_ids.append(chunk_id)
            all_embeddings.append(embedding)

        print(f"  Essay {i+1}/50 done — {len(chunks)} chunks")

    print(f"\nStoring {len(all_chunks)} chunks in ChromaDB...")

    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        collection.add(
            documents=all_chunks[i:i+batch_size],
            embeddings=all_embeddings[i:i+batch_size],
            ids=all_ids[i:i+batch_size]
        )
        print(f"  Stored batch {i//batch_size + 1}")

    print(f"\n✅ Done — {collection.count()} chunks stored in ChromaDB")

if __name__ == "__main__":
    main()