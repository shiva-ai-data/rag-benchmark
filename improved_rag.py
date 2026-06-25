import os
import chromadb
import cohere
from openai import OpenAI
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
cohere_client = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))

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

def rerank(query, chunks, top_n=3):
    response = cohere_client.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=chunks,
        top_n=top_n
    )
    reranked = [chunks[r.index] for r in response.results]
    scores = [round(r.relevance_score, 3) for r in response.results]
    return reranked, scores

def generate_answer(query, chunks):
    context = "\n\n".join(chunks)
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
    reranked_chunks, scores = rerank(query, chunks)
    print(f"Reranked scores: {scores}")
    answer = generate_answer(query, reranked_chunks)
    print(f"Answer: {answer}")
    return {
        "question": query,
        "chunks": reranked_chunks,
        "answer": answer,
        "scores": scores
    }

if __name__ == "__main__":
    questions = [
        "What does Paul Graham say about how to get startup ideas?",
        "What makes a good founder according to Paul Graham?",
        "How does Paul Graham describe the importance of writing?",
        "What advice does Paul Graham give about hiring?",
        "What does Paul Graham think about working on hard problems?"
    ]

    print("=== Improved RAG Pipeline (with Cohere reranking) ===")
    results = []
    for q in questions:
        result = ask(q)
        results.append(result)

    print("\n=== Done — 5 questions answered ===")