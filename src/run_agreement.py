"""How well do independent labellers agree on Question Time directness? (the report's Table 6 and Table B2)

Usage: python -m src.run_agreement team_labelling/labelling_sheet_Brian_shimmer.xlsx team_labelling/labelling_sheet_malavika.xlsx

The first labeller is the original label set (team_labelling/original_labels.csv, made with an AI assistant); each sheet
given on the command line adds one more labeller. For every pair of labellers it prints agreement and Cohen's kappa for the
three-way labels (D, P, E) and for the two-way decision "evasive or not". It also retrains the directness model on the pairs
that were NOT in the sheet and checks how well it agrees with each human labeller. Writes results/agreement.json.
"""

import itertools
import json
import sys

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# the same feature groups as the deployed model (see src/make_figures.py)
LEX = ["attack", "refer_prev", "thanks_start", "yesno_start", "digits", "log_a_words", "log_q_words", "q_marks", "overlap"]
SEM = ["sim_first", "sim_max", "sim_mean"]
CE = ["ce_first", "ce_max", "ce_mean"]
ORDER = {"D": 0, "P": 1, "E": 2}


def read_sheet(path: str) -> pd.Series:
    """One labeller's D/P/E labels from a returned sheet, indexed by item number. Works with a 'your_label' column."""
    items = pd.read_excel(path, sheet_name="Items").set_index("item")
    return items["your_label"].astype(str).str.strip().str.upper()


def compare(a: pd.Series, b: pd.Series) -> dict:
    """Agreement and kappa between two label series (same index)."""
    return {
        "n": int(len(a)),
        "agree_3class": round(float((a == b).mean()), 3),
        "kappa_3class": round(cohen_kappa_score(a, b), 3),
        "kappa_3class_linear_weighted": round(cohen_kappa_score(a.map(ORDER), b.map(ORDER), weights="linear"), 3),
        "agree_evasive_or_not": round(float(((a == "E") == (b == "E")).mean()), 3),
        "kappa_evasive_or_not": round(cohen_kappa_score(a == "E", b == "E"), 3),
        "direct_vs_evasive_swaps": int((((a == "D") & (b == "E")) | ((a == "E") & (b == "D"))).sum()),
    }


def main(sheets: list[str]) -> None:
    key = pd.read_csv("team_labelling/original_labels.csv").set_index("item")
    labels = pd.read_parquet("data/evasiveness_labels.parquet").reset_index(drop=True)
    feats = pd.read_parquet("data/evasiveness_label_features.parquet").reset_index(drop=True)
    feats["pair_id"], feats["label"] = labels["pair_id"].values, labels["label"].values

    people = {"annotator_1 (original labels)": key["label"]}
    for i, path in enumerate(sheets, start=2):
        people[f"annotator_{i} ({path.split('labelling_sheet_')[-1].replace('.xlsx', '')})"] = read_sheet(path)
    d = pd.DataFrame(people).dropna()
    print(f"{len(d)} items labelled by {d.shape[1]} labellers")

    out = {"pairwise": {}, "model_vs_human": {}}
    for a, b in itertools.combinations(d.columns, 2):
        out["pairwise"][f"{a} vs {b}"] = compare(d[a], d[b])
    first = d.columns[0]
    print(pd.crosstab(d[first], d[d.columns[1]], rownames=[first], colnames=[d.columns[1]]).to_string())

    # the same comparison on questions from the other side only (the case the app reports)
    same_bloc = feats.set_index("pair_id")["same_bloc"].astype(bool)
    other = ~key.loc[d.index, "pair_id"].map(same_bloc).values
    out["other_bloc_only_annotator_1_vs_2"] = compare(d.loc[other, first], d.loc[other, d.columns[1]])

    # model trained without the questions in the sheet, then compared with each human labeller
    in_sheet = feats["pair_id"].isin(key["pair_id"])
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5))
    model.fit(feats.loc[~in_sheet, LEX + SEM + CE], feats.loc[~in_sheet, "label"])
    pred = pd.Series(model.predict(feats.loc[in_sheet, LEX + SEM + CE]), index=feats.loc[in_sheet, "pair_id"].values)
    pred = pd.Series(pred.loc[key.loc[d.index, "pair_id"].values].values, index=d.index)
    for name in d.columns[1:]:
        out["model_vs_human"][name] = compare(pred, d[name])
        out["model_vs_human"][name]["always_not_evasive_agree"] = round(float((d[name] != "E").mean()), 3)

    for section in ("pairwise", "model_vs_human"):
        print(f"\n{section}")
        for name, r in out[section].items():
            print(
                f"  {name}: 3-class {r['agree_3class']:.0%} (kappa {r['kappa_3class']:.2f}), evasive-or-not {r['agree_evasive_or_not']:.0%} (kappa {r['kappa_evasive_or_not']:.2f})"
            )
    json.dump(out, open("results/agreement.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
