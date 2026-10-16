"""Test the optional AI answer on the same questions as the retrieval comparison, plus some off-topic ones.

python -m src.run_llm_eval

Needs a working AI provider (see .env.example): a local Ollama model, or an OpenAI or Anthropic key. For each question it
retrieves the best passages the way the app does (keyword search, then the cross-encoder), asks for a short answer, and records
whether the automatic checks passed, whether the model refused, and how long it took. Off-topic questions should be refused with
the exact sentence in src/llm.py. Writes results/llm_eval.csv (every answer) and results/llm_eval.json (the summary).

The checks test form (citations exist, numbers appear in the cited passages), not whether the wording is faithful, so read the
answers in results/llm_eval.csv too.
"""

import json
import statistics
import sys
import time

import pandas as pd
from sentence_transformers import CrossEncoder

sys.path.insert(0, ".")
from src import llm
from src import retrieval as R
from src.run_retrieval_eval import QUERIES, load_mp

# questions Hansard should not be able to answer, each asked of one of the MPs above
OFF_TOPIC = [
    ("Steggall, Zali", "a recipe for banana bread"),
    ("Bandt, Adam", "the weather forecast for Sydney tomorrow"),
    ("Plibersek, Tanya", "how to change a flat car tyre"),
    ("Wilkie, Andrew", "the price of bitcoin today"),
    ("Tink, Kylea", "the score of last night's football final"),
]
PASSAGES_SHOWN = 6  # the app shows the model the six best passages


def passages_for(name: str, query: str, cross_encoder, cache: dict) -> list[dict]:
    """The passages the app would send to the model for this MP and question."""
    if name not in cache:
        speeches = load_mp(name)
        ps = R.make_passages(speeches)
        cache[name] = (ps, R.build_bm25(ps), speeches["party"].mode().iloc[0] if "party" in speeches else "")
    ps, bm25, party = cache[name]
    hits = R.rerank(cross_encoder, query, R.bm25_search(ps, bm25, query, k=60), k_final=8).head(PASSAGES_SHOWN)
    return [
        {"speaker": name, "party": party, "date": pd.Timestamp(r["date"]).strftime("%d %b %Y"), "text": r["text"]}
        for _, r in hits.iterrows()
    ]


def main() -> None:
    prov = llm.provider()
    if prov is None:
        sys.exit("No AI provider is configured. Start Ollama or add a key to .env (see .env.example), then run this again.")
    print(f"Provider: {prov['name']} ({prov['model']})", flush=True)
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    cache, rows = {}, []
    for kind, questions in (("on-topic", QUERIES), ("off-topic", OFF_TOPIC)):
        for name, query in questions:
            passages = passages_for(name, query, cross_encoder, cache)
            start = time.time()
            result = llm.answer(query, passages)
            rows.append(
                {
                    "kind": kind,
                    "mp": name,
                    "question": query,
                    "provider": result["provider"],
                    "seconds": round(time.time() - start, 1),
                    "error": result["error"] or "",
                    "refused": result["text"].strip() == llm.NO_ANSWER,
                    "checks_passed": not result["error"] and not result["warnings"],
                    "warnings": "; ".join(result["warnings"]),
                    "answer": result["text"],
                }
            )
            print(
                f"[{kind}] {query}: {'REFUSED' if rows[-1]['refused'] else rows[-1]['warnings'] or 'ok'} ({rows[-1]['seconds']} s)",
                flush=True,
            )

    df = pd.DataFrame(rows)
    df.to_csv("results/llm_eval.csv", index=False)
    on, off = df[df["kind"] == "on-topic"], df[df["kind"] == "off-topic"]
    ok = df[df["error"] == ""]
    summary = {
        "provider": prov,
        "on_topic_questions": len(on),
        "on_topic_passed_every_check": int(on["checks_passed"].sum()),
        "on_topic_refused": int(on["refused"].sum()),
        "off_topic_questions": len(off),
        "off_topic_exact_refusal": int(off["refused"].sum()),
        "off_topic_not_refused_read_these_by_hand": off.loc[~off["refused"], "question"].tolist(),
        "errors": int((df["error"] != "").sum()),
        "median_seconds_per_answer": round(statistics.median(ok["seconds"]), 1) if len(ok) else None,
    }
    json.dump(summary, open("results/llm_eval.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
