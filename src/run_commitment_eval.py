"""Evaluate pledge detectors on the 90 hand-labelled sentences (gold: data/commitment_gold.txt).

The sample is stratified: 45 sentences flagged and 45 not flagged by the FIRST version of the rules (3.79% of the
33,761 sentences in the 2,500 sampled answers were flagged by that version). Population estimates therefore weight each
sentence by its stratum, whatever detector is being scored.
"""

import json
import sys
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold

sys.path.insert(0, ".")
from src.commitment import is_commitment, commitment_type

SHARE_FLAGGED_V1 = 1280 / 33761
GOLD = dict(line.split() for line in open("data/commitment_gold.txt"))
s = pd.read_parquet("data/commitment_sample.parquet")
s["gold"] = [GOLD[str(i)] for i in range(len(s))]
s["w"] = np.where(s["rule"] != "N", SHARE_FLAGGED_V1 / 45, (1 - SHARE_FLAGGED_V1) / 45)  # 'rule' = v1 flag used for sampling
s["rule_now"] = s["sent"].map(is_commitment)
s["will_only"] = np.where(s["sent"].str.contains(r"\bwill\b", case=False), "C", "N")
s["specific"] = s["sent"].map(lambda t: commitment_type(t) == "specific pledge")
print("gold distribution:", s["gold"].value_counts().to_dict())


def weighted_pr(truth, pred, w):
    """Precision and recall, weighting each sentence by how common its stratum is in the real data."""
    tp = (w * (truth & pred)).sum()
    fp = (w * (~truth & pred)).sum()
    fn = (w * (truth & ~pred)).sum()
    return tp / (tp + fp) if tp + fp else float("nan"), tp / (tp + fn) if tp + fn else float("nan")


any_gold = s["gold"].isin(["C", "H"])
out = {}
for name, col in [("contains 'will' (baseline)", "will_only"), ("rule-based patterns", "rule_now")]:
    flag = s[col] != "N"
    P, R = weighted_pr(any_gold, flag, s["w"])
    raw_p = (any_gold & flag).sum() / flag.sum()
    out[name] = {"flagged": int(flag.sum()), "raw_precision": round(raw_p, 2), "pop_precision": round(P, 2), "pop_recall": round(R, 2)}
firm_gold = s["gold"] == "C"
P, R = weighted_pr(firm_gold, s["specific"], s["w"])
out["specific-pledge detector (rules + specificity cue)"] = {
    "flagged": int(s["specific"].sum()),
    "raw_precision": round((firm_gold & s["specific"]).sum() / max(s["specific"].sum(), 1), 2),
    "raw_recall": round((firm_gold & s["specific"]).sum() / firm_gold.sum(), 2),
    "pop_precision": round(P, 2),
    "pop_recall": round(R, 2),
    "gold_specific": int(firm_gold.sum()),
}

emb = SentenceTransformer("all-MiniLM-L6-v2").encode(s["sent"].tolist(), normalize_embeddings=True, show_progress_bar=False)
y = s["gold"].values
accs, f1s = [], []
for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0).split(emb, y):
    m = LogisticRegression(max_iter=2000, class_weight="balanced", C=2.0).fit(emb[tr], y[tr])
    p = m.predict(emb[te])
    accs.append(accuracy_score(y[te], p))
    f1s.append(f1_score(y[te], p, average="macro", zero_division=0))
out["embeddings + LogReg (CV, 3-class)"] = {"accuracy": round(float(np.mean(accs)), 2), "macro_F1": round(float(np.mean(f1s)), 2)}
out["rules 3-class (unweighted)"] = {
    "accuracy": round(accuracy_score(y, s["rule_now"]), 2),
    "macro_F1": round(f1_score(y, s["rule_now"], average="macro", zero_division=0), 2),
}
print(json.dumps(out, indent=1))
print(pd.crosstab(s["gold"], s["rule_now"], rownames=["gold"], colnames=["rules now"]))
json.dump(out, open("results/commitment_eval.json", "w"), indent=1)
