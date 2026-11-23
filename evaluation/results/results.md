# Evaluation results

Configuration: embeddings=`hashing`, reranker=`lexical`, answers=`extractive`, chunk size=900 chars.
Corpus: 9 documents / 31 chunks. Questions: 30 answerable, 3 unanswerable, 2 prompt-injection probes.

## Retrieval (answerable questions)

| Mode | Recall@1 | Recall@3 | Recall@5 | MRR | DocHit@5 | Latency (ms/query) |
|---|---|---|---|---|---|---|
| dense | 0.917 | 0.983 | 0.983 | 0.983 | 1.000 | 0.41 |
| bm25 | 0.917 | 0.983 | 0.983 | 0.983 | 1.000 | 0.22 |
| hybrid | 0.917 | 0.983 | 0.983 | 0.983 | 1.000 | 0.45 |
| hybrid_rerank | 0.917 | 0.983 | 0.983 | 0.983 | 1.000 | 2.85 |

## Answer quality (hybrid + rerank, top-5)

| Metric | Score |
|---|---|
| Groundedness (heuristic, cited-source token support) | 1.000 |
| Citation accuracy (cited doc is an expected doc) | 0.844 |
| Answers containing a citation | 1.000 |
| Keyword accuracy (expected facts present) | 0.967 |
| Abstention on unanswerable questions | 0.333 |
| Prompt-injection resistance (no leak/obedience) | 1.000 |

### By question type

| Type | n | Keyword accuracy | Groundedness |
|---|---|---|---|
| comparison | 2 | 1.000 | 1.000 |
| factual | 12 | 1.000 | 1.000 |
| lookup | 2 | 1.000 | 1.000 |
| manual | 2 | 1.000 | 1.000 |
| policy | 10 | 1.000 | 1.000 |
| summary | 2 | 0.500 | 1.000 |

> Offline mode uses hashing embeddings and an *extractive* answerer, so these numbers are a conservative
> baseline on a small synthetic corpus. Re-run with `EMBEDDING_BACKEND=sbert`, `RERANKER_BACKEND=cross-encoder`
> and `LLM_PROVIDER=anthropic|openai` for production-style numbers.
