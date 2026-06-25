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
`gpt-4o-mini`). The two runs below are independent executions of the same
`python benchmark.py`:

| Metric | Run | Baseline | Improved | Δ |
|---|---|---|---|---|
| Faithfulness | 1 | 0.543 | 0.625 | ↑ 0.082 |
| Faithfulness | 2 | 0.530 | 0.665 | ↑ 0.135 |
| Answer relevancy | 1 | 0.428 | 0.506 | ↑ 0.078 |
| Answer relevancy | 2 | 0.412 | 0.504 | ↑ 0.092 |

Reranking improved **both** metrics in **both** runs — faithfulness by
+0.08 to +0.14 and answer relevancy by +0.08 to +0.09.

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
