"""Cached loaders for the app's data and models, plus two small helpers used by several screens.

Streamlit re-runs the whole script on every click, so anything slow is wrapped in a cache here: tables with
st.cache_data, and models and search indexes with st.cache_resource.
"""

import pandas as pd
import pyarrow.parquet as pq
import streamlit as st

from src import global_search as G
from src import llm
from src.stats import wilson  # noqa: F401  (re-exported for the screens)
from src.ui.config import DATA


@st.cache_data(show_spinner=False)
def load_lookup():
    """Postcode -> electorate table, and the latest speaker recorded for each electorate."""
    return pd.read_parquet(DATA / "postcodes.parquet"), pd.read_parquet(DATA / "current_mp.parquet")


@st.cache_data(show_spinner=False)
def load_pairs():
    """Question Time question/answer pairs with their directness scores (2011-2025)."""
    return pd.read_parquet(DATA / "qa_scored.parquet")


@st.cache_data(show_spinner=False)
def load_mp_list():
    """One row per MP: name, party, electorate, first and last speech, number of speeches."""
    cols = ["mp_key", "displayName", "party", "electorate", "date"]
    t = pq.read_table(DATA / "speeches.parquet", columns=cols).to_pandas()
    g = (
        t.groupby("mp_key")
        .agg(
            name=("displayName", "last"),
            party=("party", "last"),
            electorate=("electorate", "last"),
            first=("date", "min"),
            last=("date", "max"),
            n=("date", "size"),
        )
        .reset_index()
    )
    return g.sort_values("name")


@st.cache_data(show_spinner=False, max_entries=8)
def load_speeches(mp_key: str) -> pd.DataFrame:
    """All speeches by one MP. The parquet file is sorted by MP, so this filter only reads a few row groups."""
    t = pq.read_table(DATA / "speeches.parquet", filters=[("mp_key", "=", mp_key)]).to_pandas()
    return t.sort_values("date").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_topic_counts() -> pd.DataFrame:
    """Speeches per year, party and subject across the whole House (speeches without a subject are left out)."""
    t = pq.read_table(DATA / "speeches.parquet", columns=["year", "party", "topic"]).to_pandas()
    t = t[t["topic"] >= 0]
    return t.groupby(["year", "party", "topic"]).size().reset_index(name="speeches")


@st.cache_data(show_spinner=False)
def load_minister_directness(min_answers: int = 40) -> pd.DataFrame:
    """For every MP with enough Question Time answers to the other side: share scored 'not direct', with a 95% range.

    Same rules as the single-MP screen and results/minister_directness.csv: questions to the Speaker and questions from the
    minister's own side are left out.
    """
    pairs = load_pairs()
    opp = pairs[~pairs["a_is_chair"] & ~pairs["same_bloc"]]
    rows = []
    for key, g in opp.groupby("a_key"):
        if len(g) < min_answers:
            continue
        k = int((g["pred"] == "E").sum())
        low, high = wilson(k, len(g))
        rows.append(
            {
                "mp_key": key,
                "name": g["a_name"].iloc[-1],
                "party": g["a_party"].mode().iloc[0],
                "answers": len(g),
                "share": k / len(g),
                "low": low,
                "high": high,
                "first": int(g["year"].min()),
                "last": int(g["year"].max()),
            }
        )
    return pd.DataFrame(rows), float((opp["pred"] == "E").mean())


@st.cache_resource(show_spinner="Loading language models (first time only, about 15 seconds)...")
def load_models():
    """Sentence-embedding model (picks key sentences) and cross-encoder (scores how well a passage answers a query)."""
    from sentence_transformers import CrossEncoder, SentenceTransformer

    return SentenceTransformer("all-MiniLM-L6-v2"), CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


@st.cache_resource(show_spinner="Loading the search index (first time only)...")
def load_global():
    """Keyword index over every speech, used by the 'Search all MPs' screen."""
    return G.GlobalSearch()


@st.cache_data(ttl=60, show_spinner=False)
def ai_provider():
    """Which AI service is configured in .env (or None). Re-checked every minute so a new key is noticed without a restart."""
    return llm.provider()


def cite(row) -> str:
    """Citation text for a speech or passage, for example 'House of Representatives Hansard, 13 Feb 2008, p. 230'."""
    page = f", p. {int(row['page'])}" if pd.notna(row.get("page")) else ""
    return f"House of Representatives Hansard, {pd.Timestamp(row['date']).strftime('%d %b %Y')}{page}"
