import os
from dotenv import load_dotenv

load_dotenv()

print("\n=== Testing all APIs ===\n")

# Test 1 — OpenAI Embeddings
print("1. Testing OpenAI embeddings...")
try:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input="This is a test sentence."
    )
    vector = response.data[0].embedding
    print(f"   ✅ Success — got vector of {len(vector)} numbers\n")
except Exception as e:
    print(f"   ❌ Failed — {e}\n")

# Test 2 — Groq
print("2. Testing Groq...")
try:
    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": "Say hello in one word."}]
    )
    print(f"   ✅ Success — Groq says: {response.choices[0].message.content}\n")
except Exception as e:
    print(f"   ❌ Failed — {e}\n")

# Test 3 — Cohere Rerank
print("3. Testing Cohere rerank...")
try:
    import cohere
    client = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
    response = client.rerank(
        model="rerank-english-v3.0",
        query="What is the best way to start a startup?",
        documents=[
            "Focus on solving a real problem people have.",
            "The weather today is sunny and warm.",
            "Startups succeed by finding product market fit early."
        ]
    )
    top = response.results[0]
    print(f"   ✅ Success — top result index: {top.index}, score: {round(top.relevance_score, 3)}\n")
except Exception as e:
    print(f"   ❌ Failed — {e}\n")

# Test 4 — ChromaDB
print("4. Testing ChromaDB...")
try:
    import chromadb
    client = chromadb.Client()
    collection = client.create_collection("test")
    collection.add(
        documents=["This is a test document"],
        ids=["doc1"]
    )
    results = collection.query(query_texts=["test"], n_results=1)
    print(f"   ✅ Success — ChromaDB returned: {results['documents'][0][0]}\n")
except Exception as e:
    print(f"   ❌ Failed — {e}\n")

print("=== Done ===\n")
