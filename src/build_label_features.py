"""Compute lexical, embedding and cross-encoder features for the 120 labelled pairs (cached to data/)."""

import sys
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder

sys.path.insert(0, ".")
from src.evasion import lexical_features, overlap_feature, semantic_features, cross_encoder_features

df = pd.read_parquet("data/evasiveness_labels.parquet").reset_index(drop=True)
y = df["label"].values
# Three feature families: wording cues, embedding similarity, cross-encoder relevance.
lex = lexical_features(df)
lex["overlap"] = overlap_feature(df)
sem = semantic_features(df, SentenceTransformer("all-MiniLM-L6-v2"))
ce = cross_encoder_features(df, CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2"))
X = pd.concat([lex, sem, ce], axis=1)
X.to_parquet("data/evasiveness_label_features.parquet")
