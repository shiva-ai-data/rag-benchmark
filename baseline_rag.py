import os
import chromadb
from openai import OpenAI
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection("paul_graham")

def get_embedding(text):
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def retrieve(query, n_results=10):
    embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[embedding],
        n_results=n_results
    )
    return results["documents"][0]

def generate_answer(query, chunks):
    context = "\n\n".join(chunks[:3])
    prompt = f"""Answer the question based only on the context below.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {query}

Answer:"""

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def ask(query):
    print(f"\nQuestion: {query}")
    chunks = retrieve(query)
    print(f"Retrieved {len(chunks)} chunks")
    answer = generate_answer(query, chunks)
    print(f"Answer: {answer}")
    return {"question": query, "chunks": chunks, "answer": answer}

if __name__ == "__main__":
    questions = [
        "What does Paul Graham say about how to get startup ideas?",
        "What makes a good founder according to Paul Graham?",
        "How does Paul Graham describe the importance of writing?",
        "What advice does Paul Graham give about hiring?",
        "What does Paul Graham think about working on hard problems?"
    ]

    print("=== Baseline RAG Pipeline (no reranking) ===")
    results = []
    for q in questions:
        result = ask(q)
        results.append(result)

    print("\n=== Done — 5 questions answered ===")