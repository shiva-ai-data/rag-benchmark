import os
import json
import chromadb
import cohere
from openai import OpenAI
from groq import Groq
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

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
    return [chunks[r.index] for r in response.results]

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

questions = [
    "What makes a good founder according to Paul Graham?",
    "How does Paul Graham describe the importance of writing?",
    "What does Paul Graham think about working on hard problems?",
    "What does Paul Graham say about consistency and habits?",
    "How does Paul Graham define wealth?"
]

def build_dataset(use_rerank=False):
    data = {"question": [], "answer": [], "contexts": [], "reference": []}
    for q in questions:
        chunks = retrieve(q)
        if use_rerank:
            chunks = rerank(q, chunks)
        else:
            chunks = chunks[:3]
        answer = generate_answer(q, chunks)
        data["question"].append(q)
        data["answer"].append(answer)
        data["contexts"].append(chunks)
        data["reference"].append(answer)
    return Dataset.from_dict(data)

def run_evaluation(dataset, label):
    print(f"\nRunning RAGAS evaluation — {label}...")
    llm = LangchainLLMWrapper(ChatOpenAI(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY")
    ))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY")
    ))
    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=llm,
        embeddings=embeddings
    )
    return results

if __name__ == "__main__":
    print("Building baseline dataset...")
    baseline_dataset = build_dataset(use_rerank=False)

    print("Building improved dataset...")
    improved_dataset = build_dataset(use_rerank=True)

    baseline_results = run_evaluation(baseline_dataset, "Baseline")
    improved_results = run_evaluation(improved_dataset, "Improved (with reranking)")

    print("\n========================================")
    print("         FINAL BENCHMARK RESULTS        ")
    print("========================================")
    print(f"{'Metric':<25} {'Baseline':>10} {'Improved':>10} {'Delta':>10}")
    print("-" * 55)

    b = baseline_results
    r = improved_results

    for metric in ["faithfulness", "answer_relevancy"]:
        b_score = round(b[metric], 3)
        r_score = round(r[metric], 3)
        delta = round(r_score - b_score, 3)
        arrow = "↑" if delta > 0 else "↓"
        print(f"{metric:<25} {b_score:>10} {r_score:>10} {arrow} {abs(delta):>7}")

    print("========================================")
    print("\n✅ Evaluation complete — these are your LinkedIn numbers")