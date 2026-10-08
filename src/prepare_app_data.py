"""Build the data files the web app reads (run once; needs models/ from src/train_lda.py)."""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src import topic_worker

RAW = "data/corpus_1998_to_2025.parquet"
OUT = "app/data"


def main():
    """Clean the corpus, tag every speech with a topic, and write the files the web app reads to app/data/."""
    from multiprocessing import Pool

    os.makedirs(OUT, exist_ok=True)
    cols = [
        "date",
        "displayName",
        "page.no",
        "speech_no",
        "electorate",
        "partyAbbrev",
        "uniqueID",
        "body",
        "member",
        "interject",
        "question",
        "answer",
        "q_in_writing",
    ]
    df = pd.read_parquet(RAW, columns=cols)
    # member is blank (NaN) for MPs elected from 2022 who are missing from the dataset's validation list,
    # so keep everything except member == "0"
    df = df[
        df["electorate"].notna()
        & (df["member"] != "0")
        & (df["interject"] != "1")
        & df["body"].notna()
        & (df["body"].str.strip().str.len() > 30)
    ].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["words"] = df["body"].str.split().str.len()
    df = df.rename(columns={"page.no": "page", "partyAbbrev": "party"})
    df["party"] = df["party"].astype(str)
    df["is_question"] = df["question"] == "1"
    df["is_answer"] = df["answer"] == "1"
    df = df.drop(columns=["member", "interject", "question", "answer", "q_in_writing"])
    # uniqueID is blank for MPs elected from 2022, so fall back to the display name as the MP key
    df["mp_key"] = df["uniqueID"].fillna(df["displayName"])
    df = df.sort_values(["mp_key", "date"]).reset_index(drop=True)
    df["speech_id"] = np.arange(len(df))
    print("speeches:", len(df), flush=True)

    idx = np.where(df["words"].values >= 80)[0]
    print("inferring topics for", len(idx), "speeches on 10 processes ...", flush=True)
    bodies = df["body"].values[idx]
    chunks = [bodies[i : i + 2000].tolist() for i in range(0, len(bodies), 2000)]
    res = []
    with Pool(10, initializer=topic_worker.init) as pool:
        for n, r in enumerate(pool.imap(topic_worker.infer, chunks), 1):
            res.extend(r)
            if n % 10 == 0:
                print(n * 2000, flush=True)
    topic = np.full(len(df), -1, dtype=np.int8)
    topic_p = np.zeros(len(df), dtype=np.float32)
    topic[idx] = [k for k, _ in res]
    topic_p[idx] = [p for _, p in res]
    df["topic"] = topic
    df["topic_p"] = topic_p
    df.to_parquet(f"{OUT}/speeches.parquet", index=False, row_group_size=10000)

    pc = pd.read_csv("data/postcode_electorate.csv", dtype=str)[["postcode", "locality", "state", "electorate"]].dropna(
        subset=["electorate"]
    )
    pc = pc.drop_duplicates(["postcode", "electorate"])
    # drop PO-box / large-business ranges (1xxx, 9xxx, 02xx): they are not home addresses and the source maps them to arbitrary seats
    pc = pc[~pc["postcode"].str.match(r"^(1|9|02)")]
    pc.to_parquet(f"{OUT}/postcodes.parquet", index=False)

    last = df.sort_values("date").groupby("electorate").tail(1)[["electorate", "displayName", "mp_key", "party", "date"]]
    last.rename(columns={"date": "last_seen"}).to_parquet(f"{OUT}/current_mp.parquet", index=False)

    if os.path.exists("data/qa_pairs_scores.parquet"):
        import src.merge_scores  # noqa: F401  (writes app/data/qa_scored.parquet)
    else:
        print("qa_pairs_scores.parquet not found yet - run python -m src.merge_scores after src/score_pairs.py finishes")
    print("done")


if __name__ == "__main__":
    main()
