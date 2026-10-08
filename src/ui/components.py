"""Small reusable pieces of the interface: the how-it-works cards, example buttons, quote cards, CSV export, AI answer box."""

import html

import pandas as pd
import streamlit as st

from src import llm
from src.ui.config import HANSARD_HOME
from src.ui.data import ai_provider, cite

esc = html.escape  # every piece of text taken from the data is escaped before it goes into HTML

CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1180px;}
h1, h2, h3 {letter-spacing: -0.01em;}
.mp-card {background:#f6f3ec; border:1px solid #e3dece; border-radius:12px; padding:1rem 1.25rem; color:#1c2530;}
.mp-card h3 {margin:0 0 .25rem 0; color:#16304f;}
.cite {color:#5b6b7b; font-size:0.85rem;}
.quote {border-left:3px solid #c9a227; padding:.25rem .9rem; margin:.4rem 0; color:#1c2530;}
.step {margin-bottom:.5rem; background:#f6f3ec; border:1px solid #e3dece; border-radius:12px; padding:1rem 1.1rem; color:#1c2530; height:100%;}
.step b {color:#16304f;}
.step .n {display:inline-block; width:1.7rem; height:1.7rem; line-height:1.7rem; text-align:center; border-radius:50%;
          background:#efdfa0; font-weight:700; margin-right:.5rem;}
.ai-box {background:#eef3f8; border:1px solid #cfdceb; border-radius:12px; padding:.9rem 1.1rem; color:#1c2530; margin:.5rem 0 1rem 0;}
.foot {color:#6b7a8b; font-size:0.8rem; margin-top:2rem; border-top:1px solid #e3dece; padding-top:.6rem;}
</style>
"""


def apply_style():
    """Inject the app's small stylesheet."""
    st.markdown(CSS, unsafe_allow_html=True)


def footer(extra: str = ""):
    """Disclaimer shown at the bottom of every screen."""
    st.markdown(
        f"<div class='foot'>Parlianote is a student project and is not affiliated with the Parliament of Australia. {extra}</div>",
        unsafe_allow_html=True,
    )


def quote_card(text: str, source_html: str):
    """A quoted sentence with its source underneath. `source_html` must already be escaped."""
    st.markdown(f"<div class='quote'>{esc(text)}<div class='cite'>{source_html}</div></div>", unsafe_allow_html=True)


def cite_line(row) -> str:
    """HTML source line for a passage: citation, speech number and a link to the official Hansard site."""
    speech_no = f", speech no. {int(row['speech_no'])}" if pd.notna(row.get("speech_no")) else ""
    return (
        f"<span class='cite'>{esc(cite(row))}{speech_no} &middot; "
        f"<a href='{HANSARD_HOME}' target='_blank'>Check on the official Hansard site</a></span>"
    )


def how_it_works():
    """Three-step explainer shown on the home screens."""
    steps = [
        ("1", "Enter your postcode", "We find your local MP, the person who represents your area in Parliament."),
        ("2", "See what they say", "Browse their topics, search any issue, and read their own words."),
        ("3", "Check the source", "Every quote shows its date and page in the official record, so you can verify it."),
    ]
    for col, (n, title, text) in zip(st.columns(3), steps):
        col.markdown(f"<div class='step'><span class='n'>{n}</span><b>{title}</b><br>{text}</div>", unsafe_allow_html=True)
    st.write("")


def _set_state(**values):
    for key, value in values.items():
        st.session_state[key] = value


def example_buttons(label: str, items, key_prefix: str, target_key: str, fmt=lambda item: item):
    """A row of buttons; clicking one fills the input widget stored under `target_key` with that example."""
    st.caption(label)
    for col, item in zip(st.columns(len(items)), items):
        value = item[0] if isinstance(item, tuple) else item
        col.button(fmt(item), key=f"{key_prefix}_{value}", use_container_width=True, on_click=_set_state, kwargs={target_key: value})


def results_csv(hits: pd.DataFrame, speaker_col: str | None = None, fixed_speaker: str | None = None) -> bytes:
    """The passages shown on screen as a CSV (speaker, party, date, citation, text), for journalists and researchers."""
    out = pd.DataFrame(
        {
            "speaker": hits[speaker_col] if speaker_col else fixed_speaker,
            "party": hits["party"] if "party" in hits else "",
            "date": pd.to_datetime(hits["date"]).dt.strftime("%Y-%m-%d"),
            "citation": [cite(row) for _, row in hits.iterrows()],
            "text": hits["text"],
        }
    )
    return out.to_csv(index=False).encode("utf-8")


def ai_answer_box(kind: str, question: str, hits: pd.DataFrame, speaker_of):
    """Optional AI-written answer, grounded in the best passages. Hidden unless an API key (or Ollama) is configured.

    The answer is stored in session_state so it survives the page re-running when the user clicks elsewhere.
    """
    provider = ai_provider()
    if provider is None:
        return
    passages = [
        {
            "speaker": speaker_of(row),
            "party": row.get("party", ""),
            "date": pd.Timestamp(row["date"]).strftime("%d %b %Y"),
            "text": row["text"],
        }
        for _, row in hits.head(6).iterrows()
    ]
    key = f"ai|{kind}|{question}|{len(hits)}|{hits['date'].astype(str).iloc[0]}"
    if st.button(f"Write a short answer with AI ({provider['name']})", key="btn_" + key):
        with st.spinner("Reading the passages and writing an answer..."):
            st.session_state[key] = llm.answer(question, passages)
    result = st.session_state.get(key)
    if not result:
        st.caption("Optional: the AI reads only the passages below and writes a short answer that cites them.")
        return
    if result["error"]:
        st.warning(result["error"] + " The quotes below are still correct.")
        return
    st.markdown(
        f"<div class='ai-box'><b>Short answer</b> <span class='cite'>(written by AI from the passages below, "
        f"using {esc(result['provider'])})</span></div>",
        unsafe_allow_html=True,
    )
    st.markdown(result["text"])
    if result["warnings"]:
        st.warning("Please double-check this answer against the quotes: " + " ".join(result["warnings"]))
    with st.expander("Sources for [1], [2], ..."):
        for i, p in enumerate(passages, 1):
            st.markdown(f"**[{i}]** {p['speaker']} ({p['party']}), {p['date']}: {p['text'][:220]}...")
    st.caption("AI can make mistakes. Every claim should be checked against the quotes and the official record.")
