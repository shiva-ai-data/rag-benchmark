"""Quantitative benchmark: baseline vs. reranking.

Runs the evaluation question set through both pipeline variants and scores the
answers with RAGAS. We report two reference-free metrics:

* faithfulness     — is the answer grounded in the retrieved context?
* answer_relevancy — does the answer actually address the question?

Both are computed by an LLM judge (gpt-4o-mini), so no hand-labelled ground
truth is required.
"""

import time

from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, faithfulness

from config import settings
from rag import RAGPipeline

METRICS = [faithfulness, answer_relevancy]

QUESTIONS = [
    "What makes a good founder according to Paul Graham?",
    "How does Paul Graham describe the importance of writing?",
    "What does Paul Graham think about working on hard problems?",
    "What does Paul Graham say about consistency and habits?",
    "How does Paul Graham define wealth?",
    "What does Paul Graham say about how to get startup ideas?",
    "What advice does Paul Graham give about hiring?",
    "Why does Paul Graham think schlep work matters?",
    "What does Paul Graham say about doing things that don't scale?",
    "How does Paul Graham view the relationship between cities and ambition?",
    "What does Paul Graham say about the danger of premature optimization?",
    "How does Paul Graham describe what makes an essay good?",
    "What does Paul Graham think about the role of determination in success?",
    "What does Paul Graham say about taxes and risk-taking?",
    "How does Paul Graham describe the difference between makers and managers?",
    "What does Paul Graham say about how to do great work?",
    "What does Paul Graham think about copying and imitation when learning?",
    "How does Paul Graham view the value of being earnest?",
    "What does Paul Graham say about why startups should stay small early on?",
    "How does Paul Graham describe the trap of working on the wrong thing?",
    "What does Paul Graham say about how universities should teach?",
    "What does Paul Graham think about the relationship between money and happiness?",
    "How does Paul Graham describe the importance of users in a startup?",
    "What does Paul Graham say about the value of independent thinking?",
    "How does Paul Graham view the role of luck versus skill?",
]


def build_dataset(pipeline: RAGPipeline) -> Dataset:
    """Run every question through a pipeline and shape it for RAGAS."""
    records = {"question": [], "answer": [], "contexts": []}
    for question in QUESTIONS:
        result = pipeline.answer(question)
        records["question"].append(result.question)
        records["answer"].append(result.answer)
        records["contexts"].append(result.contexts)
        time.sleep(1)  # smooth out bursts against Groq's free-tier TPM limit
    return Dataset.from_dict(records)


def score(dataset: Dataset, label: str) -> dict:
    print(f"Scoring with RAGAS — {label}...")
    judge = LangchainLLMWrapper(
        ChatOpenAI(model=settings.eval_model, api_key=settings.openai_api_key)
    )
    embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model=settings.embedding_model, api_key=settings.openai_api_key
        )
    )
    return evaluate(
        dataset=dataset, metrics=METRICS, llm=judge, embeddings=embeddings
    )


def print_table(baseline: dict, improved: dict) -> None:
    print("\n" + "=" * 56)
    print("                 BENCHMARK RESULTS")
    print("=" * 56)
    print(f"{'Metric':<20}{'Baseline':>11}{'Improved':>11}{'Delta':>11}")
    print("-" * 56)
    for metric in ("faithfulness", "answer_relevancy"):
        b, i = round(baseline[metric], 3), round(improved[metric], 3)
        delta = round(i - b, 3)
        arrow = "↑" if delta > 0 else "↓" if delta < 0 else "="
        print(f"{metric:<20}{b:>11}{i:>11}{arrow + ' ' + str(abs(delta)):>11}")
    print("=" * 56)


def main() -> None:
    settings.require("openai_api_key", "groq_api_key", "cohere_api_key")

    print("Building baseline answers...")
    baseline = build_dataset(RAGPipeline(rerank=False))

    print("Building improved answers (with reranking)...")
    improved = build_dataset(RAGPipeline(rerank=True))

    baseline_scores = score(baseline, "baseline")
    improved_scores = score(improved, "improved")

    print_table(baseline_scores, improved_scores)


if __name__ == "__main__":
    main()
