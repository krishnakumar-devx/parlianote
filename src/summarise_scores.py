"""Summaries of the scored Question Time pairs used in the report (run after merge_scores).

Writes results/scoring_summary.json, results/minister_directness.csv and results/evasive_share_by_year.csv.
Only questions from the other bloc are compared, and answers by the Speaker are excluded.
"""

import json
import sys

import pandas as pd

sys.path.insert(0, ".")
from src.stats import wilson

pairs = pd.read_parquet("app/data/qa_scored.parquet")
pairs = pairs[~pairs["a_is_chair"]]  # drop questions put to the Speaker
opp = pairs[~pairs["same_bloc"]]
fri = pairs[pairs["same_bloc"]]
out = {
    "pairs": int(len(pairs)),
    "opposition_pairs": int(len(opp)),
    "friendly_pairs": int(len(fri)),
    "pred_share_all": pairs["pred"].value_counts(normalize=True).round(3).to_dict(),
    "pred_share_opposition": opp["pred"].value_counts(normalize=True).round(3).to_dict(),
    "pred_share_friendly": fri["pred"].value_counts(normalize=True).round(3).to_dict(),
}
base = (opp["pred"] == "E").mean()
rows = []
for key, g in opp.groupby("a_key"):
    if len(g) >= 40:
        k = int((g["pred"] == "E").sum())
        lo, hi = wilson(k, len(g))
        rows.append(
            {
                "minister": g["a_name"].iloc[0],
                "party": g["a_party"].mode().iloc[0],
                "n_opposition_answers": len(g),
                "predicted_evasive": round(k / len(g), 3),
                "ci_low": round(lo, 3),
                "ci_high": round(hi, 3),
                "mean_p_evasive": round(g["p_E"].mean(), 3),
            }
        )
t = pd.DataFrame(rows).sort_values("predicted_evasive", ascending=False)
t.to_csv("results/minister_directness.csv", index=False)
by_year = opp.groupby("year").apply(lambda g: (g["pred"] == "E").mean()).round(3)
by_year.to_csv("results/evasive_share_by_year.csv")
out["baseline_evasive_opposition"] = round(float(base), 3)
out["n_ministers_ge40"] = int(len(t))
out["ministers_ci_excludes_baseline_high"] = int((t["ci_low"] > base).sum())
out["ministers_ci_excludes_baseline_low"] = int((t["ci_high"] < base).sum())
json.dump(out, open("results/scoring_summary.json", "w"), indent=1)
print(json.dumps(out, indent=1))
print(t.to_string(index=False))
print(by_year.to_string())
