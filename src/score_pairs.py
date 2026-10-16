"""Train the final directness model on the labelled pairs and score every Question Time pair (2011-2025)."""

import os
import sys
import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, ".")
from src.evasion import lexical_features, overlap_feature, semantic_features, cross_encoder_features

# Train the final model on the 120 labelled pairs (same settings as rescore.py).
FEATS = None
os.makedirs("results", exist_ok=True)
lab = pd.read_parquet("data/evasiveness_labels.parquet").reset_index(drop=True)
Xlab = pd.read_parquet("data/evasiveness_label_features.parquet")
FEATS = [c for c in Xlab.columns]  # lexical + overlap + semantic + cross-encoder (includes same_party)
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5))
clf.fit(Xlab[FEATS], lab["label"].values)
joblib.dump({"model": clf, "features": FEATS}, "results/directness_model.joblib")
print("trained on", len(lab), "labelled pairs; classes:", clf.classes_)

# Compute features for every pair in chunks of 1,000 (about 1 hour on CPU) and predict.
pairs = pd.read_parquet("data/qa_pairs.parquet").reset_index(drop=True)
bi = SentenceTransformer("all-MiniLM-L6-v2")
ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
parts = []
step = 1000
for s in range(0, len(pairs), step):
    chunk = pairs.iloc[s : s + step].reset_index(drop=True)
    lex = lexical_features(chunk)
    lex["overlap"] = overlap_feature(chunk)
    X = pd.concat([lex, semantic_features(chunk, bi), cross_encoder_features(chunk, ce)], axis=1)
    proba = clf.predict_proba(X[FEATS])
    out = pd.DataFrame(proba, columns=[f"p_{c}" for c in clf.classes_])
    out["pair_id"] = chunk["pair_id"].values
    out = pd.concat([out, X[FEATS].reset_index(drop=True)], axis=1)
    parts.append(out)
    print(f"scored {min(s + step, len(pairs))}/{len(pairs)}", flush=True)
scored = pd.concat(parts, ignore_index=True)
# The label is the most probable class.
scored["pred"] = scored[["p_D", "p_E", "p_P"]].idxmax(axis=1).str[-1]
scored.to_parquet("data/qa_pairs_scores.parquet", index=False)
print(scored["pred"].value_counts(normalize=True).round(3).to_dict())
