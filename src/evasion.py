"""Directness / evasiveness features for Question Time question-answer pairs.

Two feature families are compared in the report:
  * classical: TF-IDF text features and hand-built lexical cues
  * semantic : sentence-embedding similarity between question and answer
"""

import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.blocs import bloc

ATTACK = re.compile(
    r"\b(those opposite|members opposite|the opposition|opposition leader|leader of the opposition|"
    r"shadow (treasurer|minister)|labor party|the labor|the coalition|the liberal|the greens)\b",
    re.I,
)
REFER_PREV = re.compile(
    r"\b(previous answer|earlier answer|answers? (earlier|yesterday)|my answer|"
    r"as i (said|have said|indicated)|i refer (the )?(honourable )?(member|him|her)|i refer to)\b",
    re.I,
)
THANKS = re.compile(r"^\s*(i|we) (do )?(want to )?thank (the )?(honourable )?(member|leader|deputy|shadow|minister)", re.I)
YESNO = re.compile(r"^\s*(yes|no|i do not|i don't|i do|i will|i won't|i have not|i haven't)\b", re.I)


def first_words(text: str, n: int) -> str:
    """The first n words of a text."""
    return " ".join(str(text).split()[:n])


def chunk_words(text: str, size: int = 100, max_chunks: int = 6):
    """Split text into consecutive chunks of `size` words, at most `max_chunks` of them."""
    w = str(text).split()
    return [" ".join(w[i : i + size]) for i in range(0, max(len(w), 1), size)][:max_chunks]


def lexical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Hand-built cues from each question-answer pair: attack phrases, 'see my earlier answer', digits, lengths, same-bloc flag."""
    a100 = df["a_text"].map(lambda t: first_words(t, 100))
    out = pd.DataFrame(index=df.index)
    out["attack"] = a100.map(lambda t: len(ATTACK.findall(t)))
    out["refer_prev"] = df["a_text"].map(lambda t: int(bool(REFER_PREV.search(first_words(t, 60)))))
    out["thanks_start"] = df["a_text"].map(lambda t: int(bool(THANKS.search(t))))
    out["yesno_start"] = df["a_text"].map(lambda t: int(bool(YESNO.search(t))))
    out["digits"] = a100.map(lambda t: len(re.findall(r"\d", t)))
    out["log_a_words"] = np.log1p(df["a_words"])
    out["log_q_words"] = np.log1p(df["q_words"])
    out["same_bloc"] = (df["q_party"].map(bloc) == df["a_party"].map(bloc)).astype(int)
    out["q_marks"] = df["q_text"].str.count(r"\?")
    return out


def overlap_feature(df: pd.DataFrame) -> np.ndarray:
    """TF-IDF cosine between the question and the first 120 answer words (lexical overlap)."""
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=2).fit(
        pd.concat([df["q_text"], df["a_text"].map(lambda t: first_words(t, 120))])
    )
    q = vec.transform(df["q_text"])
    a = vec.transform(df["a_text"].map(lambda t: first_words(t, 120)))
    return np.asarray(q.multiply(a).sum(axis=1)).ravel()


def semantic_features(df: pd.DataFrame, model, batch_size: int = 64) -> pd.DataFrame:
    """Embedding similarity between question and (a) answer opening, (b) best 100-word answer chunk."""
    q_emb = model.encode(
        df["q_text"].map(lambda t: first_words(t, 120)).tolist(), batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
    )
    chunks, owner = [], []
    for i, t in enumerate(df["a_text"]):
        cs = chunk_words(t)
        chunks.extend(cs)
        owner.extend([i] * len(cs))
    c_emb = model.encode(chunks, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
    owner = np.array(owner)
    sims = np.einsum("ij,ij->i", q_emb[owner], c_emb)
    first_sim, max_sim, mean_sim = (np.zeros(len(df)) for _ in range(3))
    for i in range(len(df)):
        s = sims[owner == i]
        first_sim[i], max_sim[i], mean_sim[i] = s[0], s.max(), s.mean()
    return pd.DataFrame({"sim_first": first_sim, "sim_max": max_sim, "sim_mean": mean_sim}, index=df.index)


def cross_encoder_features(df: pd.DataFrame, ce, batch_size: int = 32) -> pd.DataFrame:
    """Relevance of each 100-word answer chunk to the question, from a cross-encoder (reads both texts jointly)."""
    pairs, owner = [], []
    for i, (q, a) in enumerate(zip(df["q_text"], df["a_text"])):
        qq = first_words(q, 120)
        for c in chunk_words(a):
            pairs.append((qq, c))
            owner.append(i)
    scores = np.asarray(ce.predict(pairs, batch_size=batch_size, show_progress_bar=False))
    owner = np.array(owner)
    first, mx, mean = (np.zeros(len(df)) for _ in range(3))
    for i in range(len(df)):
        s = scores[owner == i]
        first[i], mx[i], mean[i] = s[0], s.max(), s.mean()
    return pd.DataFrame({"ce_first": first, "ce_max": mx, "ce_mean": mean}, index=df.index)
