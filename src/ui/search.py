"""Turning what the user typed into a search, and searching one MP's speeches.

The retrieval maths lives in src/retrieval.py (one MP) and src/global_search.py (everyone). This module is the thin
layer between those and the screens: it cleans the query and wraps the search in Streamlit caches.
"""

import re

import pandas as pd
import streamlit as st

from src import retrieval as R
from src.ui.config import MAX_QUERY_CHARS
from src.ui.data import load_models, load_speeches

# Question filler we remove before searching, so "what has been said about housing?" searches for "housing".
FILLER = re.compile(
    r"\b(what|who|whom|which|has|have|had|been|was|were|is|are|did|do|does|said|say|says|saying|spoke|spoken|speak|talk|talked|"
    r"talking|about|tell|me|show|us|anyone|anybody|people|mps?|politicians?|parliament|the|of|on|in|any|ever|how|why)\b",
    re.I,
)


def tidy_query(q: str) -> str:
    """Trim, collapse repeated spaces and cap the length of the text the user typed."""
    return re.sub(r"\s+", " ", q or "").strip()[:MAX_QUERY_CHARS]


def clean_query(q: str) -> str:
    """Strip question filler; if nothing meaningful would be left, keep the original text."""
    cleaned = re.sub(r"\s+", " ", FILLER.sub(" ", q)).strip(" ?.!")
    return cleaned if len(cleaned) >= 3 else q


def usable_query(q: str) -> bool:
    """True if the text contains at least one real word (not just digits or punctuation)."""
    return len(re.findall(r"[A-Za-z]{2,}", q)) >= 1


@st.cache_resource(show_spinner=False, max_entries=4)
def build_index(mp_key: str):
    """Split an MP's speeches into passages and build a keyword (BM25) index over them. Cached per MP."""
    passages = R.make_passages(load_speeches(mp_key))
    return passages, (R.build_bm25(passages) if len(passages) else None)


def retrieve(mp_key: str, query: str, year_range=None) -> pd.DataFrame:
    """Best-matching passages from one MP: keyword search first, then a cross-encoder re-reads the candidates."""
    passages, bm25 = build_index(mp_key)
    if bm25 is None:  # an MP with no usable speeches
        return pd.DataFrame()
    candidates = R.bm25_search(passages, bm25, query, k=60, year_range=year_range)
    if candidates.empty:
        return candidates
    return R.rerank(load_models()[1], query, candidates, k_final=8)


def key_sentences(query: str, hits: pd.DataFrame):
    """Three verbatim sentences from the hits that are closest to the query: [(hit_row_index, sentence), ...]."""
    return R.extractive_summary(load_models()[0], query, hits, n=3)
