"""Attach directness scores to the Question Time pairs for the app (run after rescore/score_pairs)."""

import pandas as pd
from src.blocs import bloc, CHAIR

pairs = pd.read_parquet("data/qa_pairs.parquet")
# Speaker keys (uniqueID, or the name when the ID is blank), friendly-question flag, and answers by the Speaker.
pairs["q_key"] = pairs["q_id"].fillna(pairs["q_name"])
pairs["a_key"] = pairs["a_id"].fillna(pairs["a_name"])
pairs["same_bloc"] = pairs["q_party"].map(bloc) == pairs["a_party"].map(bloc)
pairs["a_is_chair"] = pairs["a_name"].isin(CHAIR)
# Add the model's probabilities and label to each pair, then save for the app.
sc = pd.read_parquet("data/qa_pairs_scores.parquet")[["pair_id", "p_D", "p_P", "p_E", "pred"]]
pairs.merge(sc, on="pair_id").to_parquet("app/data/qa_scored.parquet", index=False)
print("written", len(pairs))
