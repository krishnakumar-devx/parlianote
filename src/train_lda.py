"""Re-train the LDA topic model exactly as in notebooks/parlianote_analysis.ipynb (Sections 1, 3, 4)."""

import os
import sys
import pandas as pd
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel

sys.path.insert(0, ".")
from src.topics import tokenize, TOPIC_NAMES

# Same cleaning, sampling and settings as the phase-1 notebook, so the model reproduces c_v 0.4592.
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)
# Real MP speeches only. This is the phase-1 filter (member == '1'); the app data uses member != '0'.
df = pd.read_parquet("data/corpus_1998_to_2025.parquet", columns=["date", "electorate", "member", "interject", "body"])
clean = df[
    df["electorate"].notna()
    & (df["member"] == "1")
    & (df["interject"] != "1")
    & df["body"].notna()
    & (df["body"].str.strip().str.len() > 30)
].copy()
clean["date"] = pd.to_datetime(clean["date"])
clean["year"] = clean["date"].dt.year
clean["word_count"] = clean["body"].str.split().str.len()
# Speeches of 80-3000 words, then 700 per year (seed 42) so no single year dominates the topics.
substantive = clean[(clean["word_count"] >= 80) & (clean["word_count"] <= 3000)]
sample = substantive.groupby("year", group_keys=False).apply(lambda g: g.sample(n=min(len(g), 700), random_state=42)).reset_index(drop=True)
print("sample:", len(sample), flush=True)
# Tokenise, then drop very rare words (<10 speeches) and very common ones (>50% of speeches).
sample["tokens"] = sample["body"].apply(tokenize)
dic = corpora.Dictionary(sample["tokens"])
dic.filter_extremes(no_below=10, no_above=0.5)
corpus = [dic.doc2bow(t) for t in sample["tokens"]]
print("vocab:", len(dic), flush=True)
# 10 topics, 8 passes, fixed seed.
lda = LdaModel(corpus=corpus, id2word=dic, num_topics=10, passes=8, random_state=42, chunksize=2000)
for i in range(10):
    words = [w for w, _ in lda.show_topic(i, 10)]
    print(f"Topic {i} ({TOPIC_NAMES[i]}): {', '.join(words)}")
lda.save("models/lda_model.gensim")
dic.save("models/lda_dictionary.gensim")
# c_v coherence: do each topic's top words tend to appear together in the texts?
coh = CoherenceModel(model=lda, texts=sample["tokens"], dictionary=dic, coherence="c_v", processes=1).get_coherence()
print(f"Coherence (c_v): {coh:.4f}")
open("results/lda_coherence.txt", "w").write(f"{coh:.4f}\n")
