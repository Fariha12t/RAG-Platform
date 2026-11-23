"""Run the full evaluation: retrieval (Recall@K, MRR) for 4 retrieval modes + answer quality.

    python -m evaluation.run_eval                # offline defaults (hashing embeddings + extractive answers)
    LLM_PROVIDER=anthropic ANTHROPIC_API_KEY=... EMBEDDING_BACKEND=sbert RERANKER_BACKEND=cross-encoder \
        python -m evaluation.run_eval            # production-style stack

Results are written to evaluation/results/results.json and results.md.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_tmp = tempfile.mkdtemp(prefix="rag_eval_")
os.environ["DATA_DIR"] = _tmp
os.environ["DATABASE_URL"] = f"sqlite:///{_tmp}/eval.db"

from app.config import get_settings                                    # noqa: E402
from app.db import Collection, init_db, session_scope                  # noqa: E402
from app.generation.llm import NOT_FOUND                               # noqa: E402
from app.ingestion.pipeline import ingest_file                         # noqa: E402
from app.retrieval.hybrid import MODES, search                         # noqa: E402
from app.services import auth, rag                                     # noqa: E402
from evaluation.metrics import (citation_accuracy, groundedness, keyword_accuracy,  # noqa: E402
                                mean, norm, retrieval_metrics)

DOCS = ROOT / "data" / "sample_docs"
DATASET = ROOT / "evaluation" / "qa_dataset.jsonl"
OUT = ROOT / "evaluation" / "results"


def main() -> None:
    s = get_settings()
    init_db()
    questions = [json.loads(line) for line in DATASET.read_text().splitlines() if line.strip()]
    with session_scope() as db:
        admin, _ = auth.create_user(db, "eval", role="admin")
        col = Collection(name="eval", owner_id=admin.id)
        db.add(col)
        db.flush()
        for f in sorted(DOCS.iterdir()):
            ingest_file(db, col, f, f.name, admin.id)
        n_chunks = sum(d.n_chunks for d in col.documents)

        # ---- retrieval ----
        answerable = [q for q in questions if q["answerable"]]
        retrieval = {}
        for mode in MODES:
            per_q, t0 = [], time.perf_counter()
            for q in answerable:
                hits = [h.__dict__ for h in search(db, col.id, q["question"], top_k=10, mode=mode)]
                per_q.append(retrieval_metrics(hits, q["expected_docs"], q["evidence"]))
            retrieval[mode] = {k: mean(m[k] for m in per_q) for k in per_q[0]}
            retrieval[mode]["avg_latency_ms"] = round((time.perf_counter() - t0) / len(answerable) * 1000, 2)

        # ---- answers (hybrid + rerank, top-5) ----
        rows, abstain, security = [], [], []
        for q in questions:
            hits = search(db, col.id, q["question"], top_k=5, mode="hybrid_rerank")
            answer, citations, sources, res, redactions = rag.answer_question(q["question"], hits)
            if q["answerable"]:
                rows.append({"id": q["id"], "type": q["type"],
                             "groundedness": groundedness(answer, sources),
                             "citation_accuracy": citation_accuracy(citations, q["expected_docs"]),
                             "has_citation": float(bool(citations)),
                             "keyword_accuracy": keyword_accuracy(answer, q["expected_keywords"])})
            elif q["type"] == "unanswerable":
                abstain.append(float(norm(NOT_FOUND) in norm(answer)))
            else:
                leaked = any(norm(w) in norm(answer) for w in q["forbidden"])
                security.append(float(not leaked))

        answers = {k: mean(r[k] for r in rows)
                   for k in ("groundedness", "citation_accuracy", "has_citation", "keyword_accuracy")}
        answers["abstention_accuracy_on_unanswerable"] = mean(abstain)
        answers["prompt_injection_resistance"] = mean(security)
        by_type = {}
        for t in sorted({r["type"] for r in rows}):
            sub = [r for r in rows if r["type"] == t]
            by_type[t] = {"n": len(sub), "keyword_accuracy": mean(r["keyword_accuracy"] for r in sub),
                          "groundedness": mean(r["groundedness"] for r in sub)}

    report = {"config": {"embedding": s.embedding_backend, "reranker": s.reranker_backend,
                         "llm": s.llm_provider, "chunk_size": s.chunk_size, "documents": len(list(DOCS.iterdir())),
                         "chunks": n_chunks, "questions_answerable": len(answerable),
                         "questions_unanswerable": len([q for q in questions if q["type"] == "unanswerable"]),
                         "questions_security": len([q for q in questions if q["type"] == "security"])},
              "retrieval": retrieval, "answers": answers, "answers_by_question_type": by_type}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(report, indent=2))
    (OUT / "results.md").write_text(render_markdown(report))
    print(render_markdown(report))


def render_markdown(r: dict) -> str:
    c = r["config"]
    lines = ["# Evaluation results", "",
             f"Configuration: embeddings=`{c['embedding']}`, reranker=`{c['reranker']}`, answers=`{c['llm']}`, "
             f"chunk size={c['chunk_size']} chars.",
             f"Corpus: {c['documents']} documents / {c['chunks']} chunks. Questions: {c['questions_answerable']} answerable, "
             f"{c['questions_unanswerable']} unanswerable, {c['questions_security']} prompt-injection probes.", "",
             "## Retrieval (answerable questions)", "",
             "| Mode | Recall@1 | Recall@3 | Recall@5 | MRR | DocHit@5 | Latency (ms/query) |",
             "|---|---|---|---|---|---|---|"]
    for mode, m in r["retrieval"].items():
        lines.append(f"| {mode} | {m['recall@1']:.3f} | {m['recall@3']:.3f} | {m['recall@5']:.3f} | {m['mrr']:.3f} | "
                     f"{m['dochit@5']:.3f} | {m['avg_latency_ms']} |")
    a = r["answers"]
    lines += ["", "## Answer quality (hybrid + rerank, top-5)", "",
              "| Metric | Score |", "|---|---|",
              f"| Groundedness (heuristic, cited-source token support) | {a['groundedness']:.3f} |",
              f"| Citation accuracy (cited doc is an expected doc) | {a['citation_accuracy']:.3f} |",
              f"| Answers containing a citation | {a['has_citation']:.3f} |",
              f"| Keyword accuracy (expected facts present) | {a['keyword_accuracy']:.3f} |",
              f"| Abstention on unanswerable questions | {a['abstention_accuracy_on_unanswerable']:.3f} |",
              f"| Prompt-injection resistance (no leak/obedience) | {a['prompt_injection_resistance']:.3f} |",
              "", "### By question type", "", "| Type | n | Keyword accuracy | Groundedness |", "|---|---|---|---|"]
    for t, m in r["answers_by_question_type"].items():
        lines.append(f"| {t} | {m['n']} | {m['keyword_accuracy']:.3f} | {m['groundedness']:.3f} |")
    lines += ["", "> Offline mode uses hashing embeddings and an *extractive* answerer, so these numbers are a conservative",
              "> baseline on a small synthetic corpus. Re-run with `EMBEDDING_BACKEND=sbert`, `RERANKER_BACKEND=cross-encoder`",
              "> and `LLM_PROVIDER=anthropic|openai` for production-style numbers."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
