"""Small reusable pieces of the interface: page header, section titles, stat cards, MP header, quote cards, CSV export, AI answer box."""

import html

import pandas as pd
import streamlit as st

from src import llm
from src.ui.config import HANSARD_HOME, PARTY_NAMES
from src.ui.data import ai_provider, cite

esc = html.escape  # every piece of text taken from the data is escaped before it goes into HTML

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap');

:root {
  --ink:#173326; --text:#1f2a24; --muted:#5e7266; --line:#d3ddd1; --soft:#e3ebe1; --card:#ffffff;
  --accent:#1f5c3f; --accent-soft:#fbf3dc; --blue-soft:#e4eee6;
}
html, body, .stApp, .stMarkdown, button, input, textarea, select {font-family:'Inter', system-ui, sans-serif;}
.stApp {background:#eef2ec;}

/* hide Streamlit's own chrome so the page reads as one product */
[data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer {display:none !important;}
header[data-testid="stHeader"] {background:transparent; height:0;}
.block-container {max-width:1080px; padding-top:2rem; padding-bottom:3rem;}
h1, h2, h3, h4 {color:var(--ink); letter-spacing:-0.015em;}

/* header */
.hero {padding:.4rem 0 1.4rem 0;}
.brand {display:flex; align-items:center; gap:.55rem; font-weight:700; color:var(--ink); font-size:1.05rem;}
.brand .logo {width:30px; height:30px; border-radius:8px; background:#1f5c3f; color:#f1e3b0; display:grid; place-items:center;
              font-family:'Source Serif 4', serif; font-size:1.1rem;}
.brand .badge {margin-left:auto; font-weight:500; font-size:.75rem; color:var(--muted); background:var(--soft);
               border:1px solid var(--line); border-radius:999px; padding:.2rem .65rem;}
.hero h1 {font-family:'Source Serif 4', serif; font-weight:600; font-size:2.6rem; line-height:1.12; margin:1.4rem 0 .5rem 0; padding:0;}
.hero p {color:var(--muted); font-size:1.05rem; margin:0; max-width:640px;}

/* segmented controls: every horizontal radio becomes a pill switch.
   Streamlit up to ~1.4x draws options as label[data-baseweb="radio"]; newer versions as label[data-testid="stRadioOption"]. */
div[data-testid="stRadio"] div[role="radiogroup"] {display:inline-flex; flex-wrap:wrap; gap:4px; background:#dce5da;
  padding:4px; border-radius:12px;}
div[data-testid="stRadio"] :is(label[data-baseweb="radio"], label[data-testid="stRadioOption"]) {margin:0 !important;
  padding:.42rem 1rem; border-radius:9px; cursor:pointer; transition:background .15s;}
div[data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child,
div[data-testid="stRadio"] label[data-testid="stRadioOption"] > div > div:not([data-testid="stMarkdownContainer"]) {display:none;}
div[data-testid="stRadio"] :is(label[data-baseweb="radio"], label[data-testid="stRadioOption"]) p {font-size:.92rem; font-weight:500;
  color:#4b5563; margin:0;}
div[data-testid="stRadio"] :is(label[data-baseweb="radio"], label[data-testid="stRadioOption"]):hover {background:rgba(255,255,255,.55);}
div[data-testid="stRadio"] :is(label[data-baseweb="radio"], label[data-testid="stRadioOption"]):has(input:checked) {background:#fff;
  box-shadow:0 1px 3px rgba(23,51,38,.14);}
div[data-testid="stRadio"] :is(label[data-baseweb="radio"], label[data-testid="stRadioOption"]):has(input:checked) p {color:var(--ink);
  font-weight:600;}

/* cards */
div[data-testid="stVerticalBlockBorderWrapper"] {border-radius:16px; border-color:var(--line) !important;}
.hint {color:var(--muted); font-size:.88rem; margin:.15rem 0 .2rem 0;}

/* buttons: quiet pills */
.stButton button, .stDownloadButton button {border-radius:999px; border:1px solid var(--line); background:#fff; color:var(--ink);
  font-weight:500; padding:.3rem 1rem;}
.stButton button:hover, .stDownloadButton button:hover {border-color:var(--ink); color:var(--ink);}

/* three-step explainer */
.steps {display:grid; grid-template-columns:repeat(3, 1fr); gap:12px; margin:1.6rem 0 1.2rem 0;}
.step {background:transparent; border-top:2px solid var(--line); padding:.8rem .2rem 0 0;}
.step .n {font-size:.75rem; font-weight:600; color:var(--accent); letter-spacing:.06em; text-transform:uppercase;}
.step b {display:block; color:var(--ink); margin:.25rem 0 .2rem 0; font-size:1rem;}
.step span.t {color:var(--muted); font-size:.9rem;}
@media (max-width: 700px) {.steps {grid-template-columns:1fr;} .hero h1 {font-size:2rem;}}

/* MP header */
.mp {display:flex; gap:1rem; align-items:center; padding:1.1rem 1.25rem; background:#fff; border:1px solid var(--line);
     border-radius:16px; margin:.6rem 0 .4rem 0;}
.mp .avatar {flex:none; width:56px; height:56px; border-radius:50%; background:#1f5c3f; color:#fff; display:grid; place-items:center;
             font-weight:600; font-size:1.15rem;}
.mp .name {font-family:'Source Serif 4', serif; font-size:1.6rem; font-weight:600; color:var(--ink); line-height:1.15;}
.mp .meta {color:var(--muted); font-size:.92rem; margin-top:.3rem; display:flex; flex-wrap:wrap; gap:.4rem .8rem; align-items:center;}
.chip {display:inline-block; font-size:.75rem; font-weight:600; padding:.15rem .6rem; border-radius:999px; background:var(--soft);
       color:var(--ink); border:1px solid var(--line);}
.notice {background:var(--accent-soft); border:1px solid #f0e0b0; color:#6b4e00; border-radius:12px; padding:.6rem .9rem;
         font-size:.88rem; margin:.4rem 0;}

/* section titles */
.sec {margin:1.8rem 0 .6rem 0;}
.sec h4 {margin:0; padding:0; font-size:1.1rem; font-weight:600;}
.sec p {margin:.2rem 0 0 0; color:var(--muted); font-size:.88rem;}

/* stat cards */
.stats {display:grid; grid-template-columns:repeat(4, 1fr); gap:12px; margin:1rem 0 .4rem 0;}
.stat {background:#fff; border:1px solid var(--line); border-radius:14px; padding:.85rem 1rem;}
.stat .l {color:var(--muted); font-size:.8rem;}
.stat .v {color:var(--ink); font-size:1.45rem; font-weight:650; margin-top:.15rem; line-height:1.25;}
.stat .v.small {font-size:1.02rem; font-weight:600;}
@media (max-width: 700px) {.stats {grid-template-columns:repeat(2, 1fr);}}
div[data-testid="stMetric"] {background:#fff; border:1px solid var(--line); border-radius:14px; padding:.8rem 1rem;}

/* quotes and sources */
.quote {background:#fff; border:1px solid var(--line); border-left:3px solid var(--accent); border-radius:4px 12px 12px 4px;
        padding:.85rem 1.1rem; margin:.5rem 0; color:var(--text); font-family:'Source Serif 4', serif; font-size:1.05rem; line-height:1.55;}
.quote .cite {font-family:'Inter', sans-serif; margin-top:.45rem;}
.cite {color:var(--muted); font-size:.82rem;}
.cite a {color:var(--muted);}

/* who spoke */
.who {display:flex; align-items:center; gap:.8rem; padding:.55rem 0; border-bottom:1px solid var(--line);}
.who .nm {font-weight:600; color:var(--ink); min-width:0;}
.who .sub {color:var(--muted); font-size:.85rem;}
.who .bar {margin-left:auto; flex:none; width:140px; height:6px; background:var(--soft); border-radius:99px; overflow:hidden;}
.who .bar i {display:block; height:100%; background:#1f5c3f; border-radius:99px;}
.who .cnt {flex:none; width:70px; text-align:right; color:var(--muted); font-size:.82rem;}

div[data-testid="stExpander"] details {border-radius:12px; border-color:var(--line); background:#fff;}
.ai-box {background:var(--blue-soft); border:1px solid #c9dbcd; border-radius:12px; padding:.7rem 1rem; color:var(--text); margin:.5rem 0 .4rem 0;}
.foot {color:var(--muted); font-size:.78rem; margin-top:3rem; border-top:1px solid var(--line); padding-top:.9rem; line-height:1.6;}
</style>
"""


def apply_style():
    """Inject the app's stylesheet."""
    st.markdown(CSS, unsafe_allow_html=True)


def hero(badge: str):
    """Page header: brand, one-line promise, and a small status badge (AI answers on or off)."""
    st.markdown(
        f"<div class='hero'><div class='brand'><span class='logo'>P</span>Parlianote<span class='badge'>{esc(badge)}</span></div>"
        "<h1>What your MP actually says in Parliament.</h1>"
        "<p>An MP (Member of Parliament) is the person who represents your local area. Everything here comes from Hansard, "
        "the official written record of the House of Representatives, March 1998 to July 2025, with the source for every quote.</p></div>",
        unsafe_allow_html=True,
    )


def hint(text: str):
    """A short grey line of guidance."""
    st.markdown(f"<div class='hint'>{text}</div>", unsafe_allow_html=True)


def notice(text: str):
    """A soft amber note for things the user should double-check."""
    st.markdown(f"<div class='notice'>{esc(text)}</div>", unsafe_allow_html=True)


def section(title: str, sub: str = ""):
    """A section heading with an optional one-line explanation."""
    st.markdown(f"<div class='sec'><h4>{title}</h4>{f'<p>{sub}</p>' if sub else ''}</div>", unsafe_allow_html=True)


def stat_cards(items):
    """A row of headline numbers. `items` is a list of (label, value) pairs; long text values get a smaller font."""
    cells = "".join(
        f"<div class='stat'><div class='l'>{esc(label)}</div><div class='v{' small' if len(str(value)) > 12 else ''}'>{esc(str(value))}</div></div>"
        for label, value in items
    )
    st.markdown(f"<div class='stats'>{cells}</div>", unsafe_allow_html=True)


def display_name(name: str) -> str:
    """'Steggall, Zali' -> 'Zali Steggall'."""
    if ", " in name:
        last, first = name.split(", ", 1)
        return f"{first} {last}"
    return name


def party_label(abbrev: str) -> str:
    """Full party name where we know it, otherwise the abbreviation from the record."""
    return PARTY_NAMES.get(str(abbrev), str(abbrev))


def mp_header(info):
    """The chosen MP: initials, name, party, area and how much of their record we hold."""
    name = display_name(str(info["name"]))
    initials = "".join(part[0] for part in name.split()[:1] + name.split()[-1:]).upper()
    st.markdown(
        f"<div class='mp'><div class='avatar'>{esc(initials)}</div><div>"
        f"<div class='name'>{esc(name)}</div>"
        f"<div class='meta'><span class='chip'>{esc(party_label(info['party']))}</span>"
        f"<span>Member for {esc(info['electorate'])}</span>"
        f"<span>{info['n']:,} speeches, {pd.Timestamp(info['first']).year}&ndash;{pd.Timestamp(info['last']).year}</span></div>"
        f"</div></div>",
        unsafe_allow_html=True,
    )


def speaker_rows(who: pd.DataFrame, total: int):
    """'Who spoke about this' as a ranked list with a small bar for each speaker's share of the best passages."""
    top = max(int(who["passages"].max()), 1)
    rows = "".join(
        f"<div class='who'><div><div class='nm'>{esc(display_name(r.displayName))}</div>"
        f"<div class='sub'>{esc(party_label(r.party))} &middot; {esc(r.electorate)}</div></div>"
        f"<div class='bar'><i style='width:{100 * r.passages / top:.0f}%'></i></div>"
        f"<div class='cnt'>{r.passages} of {total}</div></div>"
        for r in who.itertuples()
    )
    st.markdown(rows, unsafe_allow_html=True)


def footer(extra: str = ""):
    """Sources and disclaimer shown at the bottom of every screen."""
    st.markdown(
        "<div class='foot'><b>Always check quotes against the official record.</b><br>Source: Hansard, House of Representatives, March 1998 to July 2025, via Katz &amp; Alexander (2025), CC BY 4.0. "
        "Postcodes: Matthew Proctor. Parlianote is a student project and is not affiliated with the Parliament of Australia. "
        f"{extra}</div>",
        unsafe_allow_html=True,
    )


def quote_card(text: str, source_html: str):
    """A quoted sentence with its source underneath. `source_html` must already be escaped."""
    st.markdown(f"<div class='quote'>&ldquo;{esc(text)}&rdquo;<div class='cite'>{source_html}</div></div>", unsafe_allow_html=True)


def cite_line(row) -> str:
    """HTML source line for a passage: citation, speech number and a link to the official Hansard site."""
    speech_no = f", speech no. {int(row['speech_no'])}" if pd.notna(row.get("speech_no")) else ""
    return (
        f"<span class='cite'>{esc(cite(row))}{speech_no} &middot; "
        f"<a href='{HANSARD_HOME}' target='_blank'>Check on the official Hansard site</a></span>"
    )


FIND_STEPS = [
    ("Enter your postcode", "We find the MP who represents your area in Parliament."),
    ("See what they say", "Their main topics over time, and any issue you search for."),
    ("Check the source", "Every quote shows its date and page in the official record."),
]
SEARCH_STEPS = [
    ("Type a topic or question", "For example housing, aged care or climate change."),
    ("See who spoke about it", "MPs ranked by how much of the best-matching debate is theirs."),
    ("Read their exact words", "Key sentences copied from the record, each with its source."),
]


def how_it_works(steps=FIND_STEPS):
    """Three-step explainer shown on empty screens."""
    cells = "".join(
        f"<div class='step'><div class='n'>Step {i}</div><b>{title}</b><span class='t'>{text}</span></div>"
        for i, (title, text) in enumerate(steps, 1)
    )
    st.markdown(f"<div class='steps'>{cells}</div>", unsafe_allow_html=True)


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
    result = st.session_state.get(key)
    if not result:
        col_btn, col_txt = st.columns([1, 2])
        if col_btn.button("✦ Summarise with AI", key="btn_" + key, use_container_width=True):
            with st.spinner("Reading the passages and writing an answer..."):
                st.session_state[key] = llm.answer(question, passages)
            st.rerun()
        col_txt.markdown(
            f"<div class='hint'>Optional. {esc(provider['name'])} reads only the passages below and writes a short answer that cites them.</div>",
            unsafe_allow_html=True,
        )
        return
    if result["error"]:
        st.warning(result["error"] + " The quotes below are still correct.")
        return
    st.markdown(
        f"<div class='ai-box'><b>✦ Short answer</b> <span class='cite'>written by AI ({esc(result['provider'])}) from the passages below</span></div>",
        unsafe_allow_html=True,
    )
    st.markdown(result["text"])
    if result["warnings"]:
        st.warning("Please double-check this answer against the quotes: " + " ".join(result["warnings"]))
    with st.expander("Sources for [1], [2], ..."):
        for i, p in enumerate(passages, 1):
            st.markdown(f"**[{i}]** {p['speaker']} ({p['party']}), {p['date']}: {p['text'][:220]}...")
    st.caption("AI can make mistakes. Check every claim against the quotes and the official record.")
