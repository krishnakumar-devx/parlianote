"""Passage retrieval over one MP's speeches: BM25 -> cross-encoder re-ranking -> extractive summary.

Shared by the web app and by src/run_retrieval_eval.py so that what we evaluate is what users get.
"""

import re
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from src.commitment import split_sentences

_TOKEN = re.compile(r"[a-z0-9']+")


def tokens(text: str):
    """Lower-case word tokens used by the keyword (BM25) index."""
    return _TOKEN.findall(text.lower())


def make_passages(sp: pd.DataFrame, size: int = 110, stride: int = 90) -> pd.DataFrame:
    """Split each speech into overlapping ~110-word windows, keeping the citation fields."""
    rows = []
    for r in sp[sp["words"] >= 25].itertuples():
        w = r.body.split()
        for i in range(0, max(len(w) - 20, 1), stride):
            rows.append((r.speech_id, r.date, r.page, r.speech_no, " ".join(w[i : i + size])))
    return pd.DataFrame(rows, columns=["speech_id", "date", "page", "speech_no", "text"])


def build_bm25(ps: pd.DataFrame) -> BM25Okapi:
    """Build a keyword (BM25) index over the passages."""
    return BM25Okapi([tokens(t) for t in ps["text"]])


def bm25_search(ps, bm, query, k=60, year_range=None) -> pd.DataFrame:
    """The top-k passages by BM25 keyword score for the query, optionally within a year range."""
    scores = bm.get_scores(tokens(query))
    if year_range:
        yrs = pd.to_datetime(ps["date"]).dt.year.values
        scores = np.where((yrs >= year_range[0]) & (yrs <= year_range[1]), scores, -1)
    top = [i for i in np.argsort(-scores)[:k] if scores[i] > 0]
    out = ps.iloc[top].copy()
    out["bm25"] = scores[top]
    return out


def rerank(ce, query, cand: pd.DataFrame, k_final: int = 8) -> pd.DataFrame:
    """Re-score candidate passages with the cross-encoder and keep the best `k_final`."""
    if cand.empty:
        return cand
    cand = cand.copy()
    cand["rerank"] = ce.predict([(query, t) for t in cand["text"]], batch_size=32, show_progress_bar=False)
    return cand.sort_values("rerank", ascending=False).head(k_final).reset_index(drop=True)


def dense_search(ps, passage_emb, bi, query, k=8) -> pd.DataFrame:
    """Nearest passages by embedding similarity (used only in the retrieval comparison, not in the app)."""
    q = bi.encode([query], normalize_embeddings=True, show_progress_bar=False)
    sims = passage_emb @ q.ravel()
    top = np.argsort(-sims)[:k]
    out = ps.iloc[top].copy()
    out["dense"] = sims[top]
    return out.reset_index(drop=True)


def extractive_summary(bi, query, hits: pd.DataFrame, n: int = 3):
    """Pick the n sentences (from the retrieved passages) closest to the query; returns (hit_index, sentence)."""
    sents = [(i, s) for i, r in hits.iterrows() for s in split_sentences(r["text"])]
    if not sents:
        return []
    q = bi.encode([query], normalize_embeddings=True, show_progress_bar=False)
    e = bi.encode([s for _, s in sents], normalize_embeddings=True, show_progress_bar=False)
    sim = (e @ q.T).ravel()
    chosen, seen = [], set()
    for j in np.argsort(-sim):
        key = sents[j][1][:60]
        if key in seen:
            continue
        seen.add(key)
        chosen.append(j)
        if len(chosen) == n:
            break
    chosen.sort(key=lambda j: pd.Timestamp(hits.loc[sents[j][0], "date"]))
    return [(sents[j][0], sents[j][1]) for j in chosen]


# Cut-offs chosen from on-topic vs off-topic test queries (see report): relevant MP-level searches scored 3.8 to 6.8 at the top
# (two weak topics 1.1 to 2.0); off-topic queries scored below 0.5.
STRONG_RERANK = 3.0
WEAK_RERANK = 0.5


def match_quality(top_rerank) -> str:
    """'strong' (clear answer), 'weak' (closest passages only) or 'none' (nothing relevant found)."""
    if top_rerank is None or not np.isfinite(top_rerank) or top_rerank < WEAK_RERANK:
        return "none"
    return "strong" if top_rerank >= STRONG_RERANK else "weak"
