"""Draw the two samples that were labelled (same seeds as used for the report).

* 150 candidate Question Time pairs, ~10 per year 2011-2025; the first 120 were labelled -> data/evasiveness_labels.parquet
* 90 answer sentences (45 flagged by the pledge rules, 45 not) -> data/commitment_sample.parquet, gold labels in data/commitment_gold.txt
"""

import re
import sys
import pandas as pd

sys.path.insert(0, ".")
from src.commitment import split_sentences

# The labelled sentences were drawn using the FIRST version of the pledge rules, frozen here so the draw is reproducible.
_FIRM = re.compile(
    r"\b(we|i|the government|this government|our government|the coalition|labor|the minister)\s+"
    r"(will|shall|are going to|is going to|am going to|have committed to|has committed to|are committed to|"
    r"is committed to|am committed to|commit(s)? to|guarantee|promise|pledge|undertake)\b",
    re.I,
)
_GUAR = re.compile(r"\b(guarantee|i can assure|i give (the house|an?) (assurance|undertaking)|we will not|we won't|will not be)\b", re.I)
_HEDGE = re.compile(
    r"\b(consider(ing)?|look(ing)? (at|into)|examin(e|ing)|review(ing)?|in due course|"
    r"will be (receiving|taking) advice|take (that|it) on notice|we'll see|wait for|if elected)\b",
    re.I,
)


def is_commitment_v1(sentence):
    """First version of the pledge rules, kept only so the labelled sample can be re-drawn exactly."""
    if _FIRM.search(sentence) or _GUAR.search(sentence):
        return "H" if _HEDGE.search(sentence) else "C"
    if _HEDGE.search(sentence) and re.search(r"\bwill\b", sentence, re.I):
        return "H"
    return "N"


# Pairs: about 10 per year for 2011-2025 (seed 11). Sentences: 45 flagged and 45 not flagged by the v1 rules.
p = pd.read_parquet("data/qa_pairs.parquet")
q = p[(p["q_words"] >= 8) & (p["a_words"] >= 15)]
s = q.groupby("year", group_keys=False).apply(lambda g: g.sample(n=min(len(g), 10), random_state=11))
s = s.sample(frac=1, random_state=5).reset_index(drop=True)
s[["pair_id"]].to_csv("data/label_sample_ids.csv", index=False)

a = p[p["a_words"] >= 60]
rows = []
for _, r in a.sample(2500, random_state=3).iterrows():
    for sent in split_sentences(r["a_text"]):
        rows.append({"pair_id": r["pair_id"], "year": r["year"], "who": r["a_name"], "sent": sent, "rule": is_commitment_v1(sent)})
d = pd.DataFrame(rows)
cue = d[d["rule"] != "N"].sample(45, random_state=1)
non = d[d["rule"] == "N"].sample(45, random_state=1)
pd.concat([cue, non]).sample(frac=1, random_state=2).reset_index(drop=True).to_parquet("data/commitment_sample.parquet", index=False)
print(
    len(s), "pair candidates;", len(cue) + len(non), "sentences;", f"v1 flagged share {len(d[d['rule'] != 'N']) / len(d):.4f} of {len(d)}"
)
