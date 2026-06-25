"""Qualitative demo.

Runs a handful of questions through both pipeline variants so you can eyeball
how reranking changes the retrieved context and the final answer. For the
quantitative comparison, see ``benchmark.py``.
"""

import argparse

from rag import RAGPipeline

QUESTIONS = [
    "What does Paul Graham say about how to get startup ideas?",
    "What makes a good founder according to Paul Graham?",
    "How does Paul Graham describe the importance of writing?",
    "What advice does Paul Graham give about hiring?",
    "What does Paul Graham think about working on hard problems?",
]


def run(rerank: bool) -> None:
    label = "with reranking" if rerank else "baseline"
    print(f"=== RAG demo ({label}) ===")

    pipeline = RAGPipeline(rerank=rerank)
    for question in QUESTIONS:
        result = pipeline.answer(question)
        print(f"\nQ: {question}")
        if result.scores:
            print(f"   rerank scores: {result.scores}")
        print(f"A: {result.answer}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="run without reranking (default: reranking enabled)",
    )
    args = parser.parse_args()
    run(rerank=not args.baseline)
