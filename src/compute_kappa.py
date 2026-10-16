"""Compute inter-annotator agreement (Cohen's kappa) once teammates return their labelling sheets.

Usage: python -m src.compute_kappa team_labelling/labelling_alice.xlsx team_labelling/labelling_bob.xlsx
The first author's labels come from team_labelling/original_labels.csv.
"""

import sys
import itertools
import pandas as pd
from sklearn.metrics import cohen_kappa_score

# The first labeller is the original labels; each sheet on the command line adds another labeller.
key = pd.read_csv("team_labelling/original_labels.csv").set_index("item")["label"].rename("annotator_1 (model-assisted)")
cols = {key.name: key}
for path in sys.argv[1:]:
    df = pd.read_excel(path, sheet_name="Items").set_index("item")["your_label"].str.strip().str.upper()
    cols[path.split("labelling_")[-1].replace(".xlsx", "")] = df
# Keep items every labeller labelled, then compare each pair of labellers.
d = pd.DataFrame(cols).dropna()
print(f"{len(d)} items labelled by {d.shape[1]} annotators")
for a, b in itertools.combinations(d.columns, 2):
    print(f"kappa({a}, {b}) = {cohen_kappa_score(d[a], d[b]):.2f}   raw agreement = {(d[a] == d[b]).mean():.0%}")
