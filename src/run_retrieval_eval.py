"""Compare three retrieval methods on the same queries with pooled relevance judgements.

python -m src.run_retrieval_eval pool    -> writes results/retrieval_pool.csv (methods hidden) and prints passages to judge
python -m src.run_retrieval_eval score   -> reads data/retrieval_judgements.txt and reports precision@3 per method
"""

import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sentence_transformers import CrossEncoder, SentenceTransformer

sys.path.insert(0, ".")
from src import retrieval as R

QUERIES = [  # (MP displayName, query)
    ("Steggall, Zali", "climate change and emissions targets"),
    ("Steggall, Zali", "political donations and integrity"),
    ("Steggall, Zali", "gambling advertising reform"),
    ("Bandt, Adam", "renters and the cost of housing"),
    ("Bandt, Adam", "burning coal and gas and climate change"),
    ("Plibersek, Tanya", "paid parental leave"),
    ("Plibersek, Tanya", "funding for public schools"),
    ("Wilkie, Andrew", "poker machines and gambling harm"),
    ("Wilkie, Andrew", "Tasmania hospitals and health funding"),
    ("Tink, Kylea", "recycling and plastic waste"),
    # paraphrase queries: the MP's own wording differs from the query
    ("Steggall, Zali", "keeping politicians honest"),
    ("Bandt, Adam", "people struggling to pay their rent"),
    ("Plibersek, Tanya", "support for mums and dads with newborn babies"),
    ("Wilkie, Andrew", "protecting vulnerable people from betting addiction"),
    ("Tink, Kylea", "cutting the rubbish that ends up in landfill"),
]
K = 3


def load_mp(name):
    """All speeches by one MP, oldest first."""
    t = pq.read_table("app/data/speeches.parquet", filters=[("displayName", "=", name)]).to_pandas()
    return t.sort_values("date").reset_index(drop=True)


def pool(start=0):
    """Collect each method's top results for every query, hide which method found what, and print the pooled passages to judge."""
    bi = SentenceTransformer("all-MiniLM-L6-v2")
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    cache = {}
    rows, truth = [], []
    for qid, (name, query) in enumerate(QUERIES):
        if name not in cache:
            sp = load_mp(name)
            ps = R.make_passages(sp)
            bm = R.build_bm25(ps)
            cp = f"results/_emb_{name.split(',')[0]}.npy"
            try:
                emb = np.load(cp)
            except FileNotFoundError:
                emb = bi.encode(ps["text"].tolist(), batch_size=64, normalize_embeddings=True, show_progress_bar=False)
                np.save(cp, emb)
            cache[name] = (ps, bm, emb)
            print(f"[{name}] {len(sp)} speeches, {len(ps)} passages", flush=True)
        ps, bm, emb = cache[name]
        cand = R.bm25_search(ps, bm, query, k=60)
        runs = {
            "bm25": cand.head(K),
            "bm25+cross-encoder": R.rerank(ce, query, cand, k_final=K),
            "dense (MiniLM)": R.dense_search(ps, emb, bi, query, k=K),
        }
        seen = {}
        for m, df in runs.items():
            for rank, r in enumerate(df.itertuples(), 1):
                key = (r.speech_id, r.text[:50])
                if key not in seen:
                    seen[key] = len(seen)
                    rows.append({"qid": qid, "pid": seen[key], "text": r.text})
                truth.append({"qid": qid, "method": m, "rank": rank, "pid": seen[key]})
    pd.DataFrame(rows).to_csv("results/retrieval_pool.csv", index=False)
    pd.DataFrame(truth).to_csv("results/retrieval_pool_methods_HIDDEN.csv", index=False)
    pooled = pd.DataFrame(rows).sample(frac=1, random_state=3)
    for qid, g in pooled.groupby("qid"):
        if qid < start:
            continue
        print(f"\n##### Q{qid}: {QUERIES[qid][0]} | {QUERIES[qid][1]}")
        for r in g.itertuples():
            print(f"[{qid}.{r.pid}] {' '.join(r.text.split()[:75])}")


def score():
    """Compute precision@3 for each method from the judgements in data/retrieval_judgements.txt."""
    judg = {}
    for line in open("data/retrieval_judgements.txt"):
        k, v = line.split()
        judg[k] = int(v)
    truth = pd.read_csv("results/retrieval_pool_methods_HIDDEN.csv")
    truth["rel"] = [judg[f"{q}.{p}"] for q, p in zip(truth["qid"], truth["pid"])]
    out = truth.groupby("method")["rel"].agg(["mean", "sum", "count"]).rename(columns={"mean": "precision@3"})
    truth["kind"] = np.where(truth["qid"] < 10, "keyword", "paraphrase")
    for kind in ["keyword", "paraphrase"]:
        g = truth[truth["kind"] == kind].groupby("method")["rel"]
        out[f"{kind}_relevant"] = g.sum()
        out[f"{kind}_slots"] = g.count()
        out[f"{kind}_p3"] = g.mean()
    print(out.round(3))
    per_q = truth.groupby(["qid", "method"])["rel"].mean().unstack().round(2)
    per_q.insert(0, "query", [QUERIES[i][1] for i in per_q.index])
    print(per_q.to_string())
    out.round(3).to_csv("results/retrieval_comparison.csv")


if __name__ == "__main__":
    pool(int(sys.argv[2]) if len(sys.argv) > 2 else 0) if sys.argv[1] == "pool" else score()
