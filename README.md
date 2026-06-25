# RAG Benchmark — does reranking actually help?

A small, reproducible experiment that measures whether adding a **reranking**
step to a retrieval-augmented generation (RAG) pipeline improves answer
quality — using [RAGAS](https://docs.ragas.io/) for objective scoring instead
of vibes.

The corpus is 50 [Paul Graham essays](https://huggingface.co/datasets/sgoel9/paul_graham_essays).

## The experiment

Two pipelines, identical except for one step:

| | Baseline | Improved |
|---|---|---|
| Retrieve | top-10 chunks by vector similarity | top-10 chunks by vector similarity |
| Rerank | — | Cohere `rerank-english-v3.0` |
| Generate from | first 3 chunks | best 3 chunks after reranking |

The hypothesis: vector similarity is a coarse first pass, so reranking the
candidates before handing them to the LLM should produce answers that are
better grounded in the source text.

## Results

Scored with RAGAS on a 25-question evaluation set, with generation pinned to
`temperature=0` for reproducibility. Both metrics are reference-free (judged by
`gpt-4o-mini`). Numbers below are averaged over 2 runs, with the per-run range
in brackets:

| Metric | Baseline | Improved | Δ (range over 2 runs) |
|---|---|---|---|
| Faithfulness | ~0.54 | ~0.65 | **↑ ~0.11** (+0.08 to +0.14) |
| Answer relevancy | ~0.42 | ~0.51 | **↑ ~0.09** (+0.08 to +0.09) |

Reranking improved both grounding and relevance together, consistently across
runs. The effect only showed up cleanly after widening retrieval (top-40
candidates → rerank → top-3) and fixing the generation temperature — at top-10
candidates with a 5-question set, the difference was lost in run-to-run noise.

> The exact magnitude varies slightly between runs because the RAGAS judge
> samples internally; the *direction* is stable. Reproduce with
> `python benchmark.py`.

- **Faithfulness** — how well the answer is supported by the retrieved context.
- **Answer relevancy** — how directly the answer addresses the question.

## Architecture

```
config.py      central settings — models, paths, retrieval depth
rag.py         RAGPipeline: retrieve -> (optional rerank) -> generate
ingest.py      load essays, chunk, embed, store in ChromaDB
demo.py        qualitative side-by-side on example questions
benchmark.py   quantitative RAGAS comparison
test_apis.py   smoke test for all API keys
```

Both pipeline variants live in a single `RAGPipeline` class toggled by a
`rerank` flag, so retrieval and generation are written once and only the
experiment variable changes.

**Stack:** OpenAI (`text-embedding-3-small`) · ChromaDB · Cohere rerank ·
Groq (`llama-3.1-8b-instant`) · RAGAS.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # then add your OpenAI, Groq, and Cohere keys
python test_apis.py    # confirm all keys work
```

## Usage

```bash
python ingest.py            # build the vector store (one time)
python demo.py              # see reranked answers
python demo.py --baseline   # see baseline answers
python benchmark.py         # run the full RAGAS comparison
```

## Why these tools

- **Groq** for generation — fast and free-tier friendly for an experiment.
- **Cohere rerank** — a strong off-the-shelf cross-encoder; no model to host.
- **RAGAS** — turns "the answers feel better" into numbers you can defend.
