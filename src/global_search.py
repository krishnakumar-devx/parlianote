"""Search across every MP's speeches: TF-IDF finds candidate speeches, a cross-encoder picks the best passages.

Build once:  python -m src.global_search build      (writes app/data/global_index.joblib)
"""

import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SPEECHES = "app/data/speeches.parquet"
INDEX = "app/data/global_index.joblib"
META_COLS = ["speech_id", "displayName", "party", "electorate", "date", "year", "page", "speech_no"]


def build():
    """Build and save the TF-IDF index over every speech of 60 words or more."""
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer

    t = pq.read_table(SPEECHES, columns=META_COLS + ["body", "words"]).to_pandas()
    t = t[t["words"] >= 60].reset_index(drop=True)
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=5, max_df=0.4, dtype=np.float32, max_features=250000)
    X = vec.fit_transform(t["body"])
    meta = t[META_COLS].copy()
    joblib.dump({"vec": vec, "X": X.tocsr(), "meta": meta}, INDEX, compress=3)
    print("indexed", X.shape, "nnz", X.nnz)


class GlobalSearch:
    """Search every MP's speeches: TF-IDF finds candidate speeches, then a cross-encoder picks the best passages."""

    def __init__(self):
        import joblib

        d = joblib.load(INDEX)
        self.vec, self.X, self.meta = d["vec"], d["X"], d["meta"]
        self.years = self.meta["year"].values

    def candidates(self, query: str, year_range=None, k=60) -> pd.DataFrame:
        """The top-k speeches by TF-IDF similarity to the query, optionally within a year range."""
        q = self.vec.transform([query])
        sims = (self.X @ q.T).toarray().ravel()
        if year_range:
            sims = np.where((self.years >= year_range[0]) & (self.years <= year_range[1]), sims, 0)
        top = np.argsort(-sims)[:k]
        top = top[sims[top] > 0]
        out = self.meta.iloc[top].copy()
        out["tfidf"] = sims[top]
        return out

    def passages(self, cand: pd.DataFrame, query: str, per_speech=2, size=110, stride=90) -> pd.DataFrame:
        """Cut the candidate speeches into passages and keep the `per_speech` best ones by keyword similarity."""
        ids = cand["speech_id"].tolist()
        body = (
            pq.read_table(SPEECHES, columns=["speech_id", "body"], filters=[("speech_id", "in", ids)])
            .to_pandas()
            .set_index("speech_id")["body"]
        )
        qv = self.vec.transform([query])
        rows = []
        for r in cand.itertuples():
            w = body[r.speech_id].split()
            ps = [" ".join(w[i : i + size]) for i in range(0, max(len(w) - 20, 1), stride)]
            sc = (self.vec.transform(ps) @ qv.T).toarray().ravel()
            for j in np.argsort(-sc)[:per_speech]:
                if sc[j] > 0:
                    rows.append({**r._asdict(), "text": ps[j], "passage_score": float(sc[j])})
        return pd.DataFrame(rows).drop(columns=["Index"], errors="ignore")

    def search(self, query: str, ce, year_range=None, k_speeches=40, k_final=8):
        """Full search: candidate speeches, then passages, then cross-encoder re-ranking. Returns (passages, candidates)."""
        cand = self.candidates(query, year_range, k=k_speeches)
        if cand.empty:
            return pd.DataFrame(), cand
        ps = self.passages(cand, query)
        if ps.empty:
            return ps, cand
        ps["rerank"] = ce.predict([(query, t) for t in ps["text"]], batch_size=32, show_progress_bar=False)
        return ps.sort_values("rerank", ascending=False).reset_index(drop=True), cand


def who_spoke(hits: pd.DataFrame, n_top=5) -> pd.DataFrame:
    """Speakers ranked by how many of the strongest passages are theirs."""
    top = hits.head(20)
    g = top.groupby(["displayName", "party", "electorate"]).agg(passages=("text", "size"), best=("rerank", "max")).reset_index()
    return g.sort_values(["passages", "best"], ascending=False).head(n_top)


if __name__ == "__main__":
    if sys.argv[1:] == ["build"]:
        build()


# Cut-offs chosen from 10 on-topic and 10 off-topic test queries: on-topic had max TF-IDF similarity 0.43-0.67 and mean top-3
# cross-encoder score 3.2-8.9; off-topic had 0.00-0.32 and -9.6 to 2.8.
def match_quality(max_tfidf: float, mean_top3: float) -> str:
    """'strong', 'weak' or 'none' for a search across all MPs."""
    if not np.isfinite(mean_top3) or max_tfidf < 0.34 or mean_top3 < 1.0:
        return "none"
    return "strong" if (max_tfidf >= 0.38 and mean_top3 >= 3.0) else "weak"
