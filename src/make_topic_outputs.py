"""Topic summary table and topics-over-time figure for the report."""

import sys
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gensim.models import LdaModel

sys.path.insert(0, ".")
from src.topics import TOPIC_NAMES

sp = pd.read_parquet("app/data/speeches.parquet", columns=["year", "topic", "party"])
lda = LdaModel.load("models/lda_model.gensim")
sub = sp[sp["topic"] >= 0]
counts = sub["topic"].value_counts().sort_index()
rows = [
    {
        "topic": k,
        "name": TOPIC_NAMES[k],
        "top_words": ", ".join(w for w, _ in lda.show_topic(k, 8)),
        "speeches": int(counts[k]),
        "share": round(counts[k] / counts.sum(), 3),
    }
    for k in range(10)
]
pd.DataFrame(rows).to_csv("results/topic_summary.csv", index=False)
print(pd.DataFrame(rows)[["topic", "name", "speeches", "share"]].to_string(index=False))
print("speeches with a topic:", len(sub), "of", len(sp), "; no topic (<80 words):", (sp["topic"] < 0).sum())

share = sub.groupby(["year", "topic"]).size().unstack(fill_value=0)
share = share.div(share.sum(axis=1), axis=0)
share.columns = [TOPIC_NAMES[c] for c in share.columns]
palette = ["#16304F", "#2F5D8A", "#6C9BC4", "#C9A227", "#E4C766", "#B5483C", "#8A97A6", "#4F7F5B", "#9A6FA8", "#D9885F"]
fig, ax = plt.subplots(figsize=(9, 4.0))
ax.stackplot(share.index, share.T.values, labels=share.columns, colors=palette, linewidth=0.3, edgecolor="white")
ax.set_xlim(share.index.min(), share.index.max())
ax.set_ylim(0, 1)
ax.set_ylabel("Share of topic-labelled speeches")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig("results/fig_topics_over_time.png", dpi=200)
plt.close(fig)
print("year coverage:", share.index.min(), share.index.max())
print(share.loc[[1998, 2010, 2020, 2025]].round(2).T.to_string())
