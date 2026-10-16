"""Re-fit the directness model on the labelled pairs and re-predict every Question Time pair from cached features.

score_pairs.py computes the (slow) embedding and cross-encoder features once. This script lets us change the final
classifier or the friendly-question definition (src/blocs.py) without recomputing them.
"""

import os
import sys
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, ".")
from src.blocs import bloc

os.makedirs("results", exist_ok=True)
# The 120 labelled pairs and their cached features (from build_label_features.py).
lab = pd.read_parquet("data/evasiveness_labels.parquet").reset_index(drop=True)
Xlab = pd.read_parquet("data/evasiveness_label_features.parquet").reset_index(drop=True)
FEATS = list(Xlab.columns)

sc = pd.read_parquet("data/qa_pairs_scores.parquet")
pairs = pd.read_parquet("data/qa_pairs.parquet").set_index("pair_id")
# Friendly question = asker and answerer are in the same bloc (Liberal and National count as one).
sc["same_bloc"] = (pairs.loc[sc["pair_id"], "q_party"].map(bloc).values == pairs.loc[sc["pair_id"], "a_party"].map(bloc).values).astype(int)
# Final model: standardised features, then class-weighted logistic regression.
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5))
clf.fit(Xlab[FEATS], lab["label"].values)
joblib.dump({"model": clf, "features": FEATS}, "results/directness_model.joblib")
# One probability each for Direct, Partial and Evasive; the largest is the label.
proba = clf.predict_proba(sc[FEATS])
for i, c in enumerate(clf.classes_):
    sc[f"p_{c}"] = proba[:, i]
sc["pred"] = sc[["p_D", "p_E", "p_P"]].idxmax(axis=1).str[-1]
sc.to_parquet("data/qa_pairs_scores.parquet", index=False)
print("re-scored", len(sc), sc["pred"].value_counts(normalize=True).round(3).to_dict())
