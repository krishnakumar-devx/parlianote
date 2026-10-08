"""Recompute the evaluation tables from cached features and draw the report figures."""

import sys
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

sys.path.insert(0, ".")
from src.evasion import first_words

NAVY, GOLD, GREY, LIGHT = "#16304F", "#C9A227", "#8A97A6", "#F6F3EC"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

lab = pd.read_parquet("data/evasiveness_labels.parquet").reset_index(drop=True)
X = pd.read_parquet("data/evasiveness_label_features.parquet").reset_index(drop=True)
X["a_open"] = lab["a_text"].map(lambda t: first_words(t, 120))
y = lab["label"].values

LEX = ["attack", "refer_prev", "thanks_start", "yesno_start", "digits", "log_a_words", "log_q_words", "q_marks", "overlap"]
SEM = ["sim_first", "sim_max", "sim_mean"]
CE = ["ce_first", "ce_max", "ce_mean"]
lr = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5))
tfidf = lambda: Pipeline(
    [
        ("t", ColumnTransformer([("t", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), "a_open")])),
        ("c", LinearSVC(C=0.5, class_weight="balanced")),
    ]
)
MODELS = {
    "Majority class": (DummyClassifier(strategy="most_frequent"), LEX),
    "TF-IDF + SVM": (tfidf(), ["a_open"]),
    "Lexical cues + LR": (lr(), LEX),
    "Embedding similarity + LR": (lr(), SEM),
    "Cross-encoder + LR": (lr(), CE),
    "All features + LR": (lr(), LEX + SEM + CE),
}
same = (X["same_bloc"] == 1).values
SUBSETS = {f"All pairs (n={len(X)})": np.ones(len(X), bool), f"Other-bloc questions (n={int((~same).sum())})": ~same}

rows = []
for sname, mask in SUBSETS.items():
    Xs, ys = X[mask].reset_index(drop=True), y[mask]
    for task in ["3-class", "evasive vs not"]:
        yt = ys if task == "3-class" else np.where(ys == "E", "E", "notE")
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=1)
        for mname, (clf, cols) in MODELS.items():
            acc, mf = [], []
            for tr, te in cv.split(Xs, yt):
                m = clone(clf).fit(Xs.iloc[tr][cols], yt[tr])
                p = m.predict(Xs.iloc[te][cols])
                acc.append(accuracy_score(yt[te], p))
                mf.append(f1_score(yt[te], p, average="macro", zero_division=0))
            rows.append(
                {"subset": sname, "task": task, "model": mname, "accuracy": np.mean(acc), "macro_F1": np.mean(mf), "sd_acc": np.std(acc)}
            )
res = pd.DataFrame(rows).round(3)
res.to_csv("results/model_comparison.csv", index=False)
print(res.to_string(index=False))

# ---- Figure: model comparison
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
for ax, task, metric in [(axes[0], "3-class", "macro_F1"), (axes[1], "evasive vs not", "accuracy")]:
    d = res[res["task"] == task]
    names = list(MODELS.keys())
    ypos = np.arange(len(names))
    for k, (sname, col) in enumerate(zip(SUBSETS, [GREY, NAVY])):
        v = [d[(d["subset"] == sname) & (d["model"] == n)][metric].iloc[0] for n in names]
        ax.barh(ypos + (k - 0.5) * 0.38, v, height=0.36, color=col, label=sname)
    ax.set_yticks(ypos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_title("Macro-F1, 3 classes (D/P/E)" if task == "3-class" else "Accuracy, evasive vs not", fontsize=10, color=NAVY, loc="left")
    ax.grid(axis="x", alpha=0.25)
axes[0].legend(loc="lower right", frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig("results/fig_model_comparison.png", dpi=200)
plt.close(fig)

# ---- Figure: labels by question type
ct = pd.crosstab(np.where(same, "Same bloc\n(friendly)", "Different bloc\n(opposition / cross-bench)"), y)[["D", "P", "E"]]
fig, ax = plt.subplots(figsize=(6.2, 3.4))
left = np.zeros(len(ct))
for col, c, name in [("D", NAVY, "Direct"), ("P", GOLD, "Partial"), ("E", "#B5483C", "Evasive")]:
    ax.barh(ct.index, ct[col], left=left, color=c, label=name)
    for i, v in enumerate(ct[col]):
        if v > 3:
            ax.text(left[i] + v / 2, i, str(v), ha="center", va="center", color="white", fontsize=9)
    left += ct[col].values
ax.set_xlabel("Hand-labelled pairs")
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
fig.tight_layout()
fig.savefig("results/fig_labels_by_question_type.png", dpi=200)
plt.close(fig)

# ---- Figure: issue distribution (zero-shot classifier, notebook Section 5)
issues = {
    "Indigenous Affairs*": 1412,
    "Social Services & Welfare": 968,
    "Environment & Climate": 785,
    "Infrastructure & Transport": 764,
    "Health": 728,
    "Economy & Tax": 703,
    "Immigration": 543,
    "Education": 528,
    "Agriculture & Regional": 517,
    "Justice & Law": 517,
    "Defence & Security": 413,
    "Housing": 308,
}
fig, ax = plt.subplots(figsize=(6.2, 3.8))
ax.barh(list(issues.keys()), list(issues.values()), color=NAVY, height=0.62)
ax.invert_yaxis()
ax.set_xlabel("Confidently classified speeches (sample of 19,600)")
for i, v in enumerate(issues.values()):
    ax.text(v + 15, i, str(v), va="center", fontsize=8, color="#4B5A6B")
fig.tight_layout()
fig.savefig("results/fig_issue_distribution.png", dpi=200)
plt.close(fig)

# ---- Figure: system pipeline
fig, ax = plt.subplots(figsize=(10, 4.6))
ax.set_xlim(0, 100)
ax.set_ylim(-3, 44)
ax.axis("off")


def box(x, y, w, h, text, fc=LIGHT, ec=NAVY, tc=NAVY, bold=False):
    """Draw a rounded box with centred text on the system diagram."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.6, color=tc, fontweight="bold" if bold else "normal")


def arrow(x1, y1, x2, y2):
    """Draw an arrow between two points on the system diagram."""
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10, color=GOLD, lw=1.6))


box(1, 18, 15, 9, "Hansard 1998-2025\n(Parquet, 648k rows)", bold=True)
box(22, 18, 15, 9, "Clean + validate\n371k MP speeches")
arrow(16.5, 22.5, 21.5, 22.5)
box(44, 34, 23, 8, "Postcode -> electorate -> MP\n(AEC-based lookup)")
box(44, 22, 23, 8, "LDA topics (10) over\nall speeches, by year")
box(44, 10, 23, 8, "Q&A pairing (2011+) ->\ndirectness score + pledges")
box(44, 0, 23, 8, "BM25 + cross-encoder search\n-> extractive summary")
for yy in [38, 26, 14, 4]:
    arrow(37.7, 22.5, 43.5, yy)
    arrow(67.5, yy, 74.5, 22.5)
box(75, 17, 23, 11, "Parlianote web app\n(Streamlit)\nevery output cites its\nHansard source", fc=NAVY, tc="white", bold=True)
fig.tight_layout()
fig.savefig("results/fig_pipeline.png", dpi=200)
plt.close(fig)
print("figures written")
