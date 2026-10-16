"""Pair each Question Time question with the answer that follows it (same sitting day)."""

import pandas as pd
import numpy as np

RAW = "data/corpus_1998_to_2025.parquet"
OUT = "data/qa_pairs.parquet"

cols = [
    "date",
    "displayName",
    "order",
    "speech_no",
    "page.no",
    "electorate",
    "partyAbbrev",
    "uniqueID",
    "body",
    "question",
    "answer",
    "interject",
    "q_in_writing",
    "fedchamb_flag",
]
df = pd.read_parquet(RAW, columns=cols)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["date", "order"]).reset_index(drop=True)

# only oral Question Time style rows (question/answer flagged, not written questions on notice)
qa = df[((df["question"] == "1") | (df["answer"] == "1")) & (df["q_in_writing"] != "1")].copy()
qa = qa[qa["interject"] != "1"].reset_index(drop=True)

# Walk each sitting day in order: pair each question with the next answer before the next question.
pairs = []
for date, g in qa.groupby("date", sort=True):
    g = g.sort_values("order").reset_index(drop=True)
    i = 0
    n = len(g)
    while i < n:
        row = g.iloc[i]
        if row["question"] == "1":
            # find the next answer row before the next question row
            j = i + 1
            while j < n and g.iloc[j]["question"] != "1" and g.iloc[j]["answer"] != "1":
                j += 1
            if j < n and g.iloc[j]["answer"] == "1":
                a = g.iloc[j]
                pairs.append(
                    {
                        "date": date,
                        "year": date.year,
                        "q_name": row["displayName"],
                        "q_id": row["uniqueID"],
                        "q_electorate": row["electorate"],
                        "q_party": row["partyAbbrev"],
                        "q_text": row["body"],
                        "q_page": row["page.no"],
                        "q_speech_no": row["speech_no"],
                        "a_name": a["displayName"],
                        "a_id": a["uniqueID"],
                        "a_electorate": a["electorate"],
                        "a_party": a["partyAbbrev"],
                        "a_text": a["body"],
                        "a_page": a["page.no"],
                        "a_speech_no": a["speech_no"],
                    }
                )
                i = j + 1
                continue
        i += 1

# Word counts and a running ID for each pair.
p = pd.DataFrame(pairs)
p["q_words"] = p["q_text"].str.split().str.len()
p["a_words"] = p["a_text"].str.split().str.len()
p["pair_id"] = np.arange(len(p))
p.to_parquet(OUT, index=False)

print("pairs:", len(p))
print("years:", p["year"].min(), "-", p["year"].max())
print(p[["q_words", "a_words"]].describe())
print("\nTop answerers:\n", p["a_name"].value_counts().head(10))
print("\nTop askers:\n", p["q_name"].value_counts().head(10))
print("\nPairs per year:\n", p.groupby("year").size().to_string())
