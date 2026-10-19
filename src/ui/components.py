"""Small reusable pieces of the interface: page header, section titles, stat cards, MP header, quote cards, CSV export, AI answer box."""

import html
import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st

from src import llm
from src.blocs import bloc
from src.ui.config import BLOC_COLOURS, HANSARD_HOME, PARTY_NAMES
from src.ui.data import ai_provider, cite

esc = html.escape  # every piece of text taken from the data is escaped before it goes into HTML

# Each MP's ID on the Parliament's website (built by scripts/build_photo_ids.py), for their official portrait. The visitor's
# browser loads it from aph.gov.au. Photos: Parliament of Australia, CC BY-NC-ND 4.0, which allows sharing a photo "in whole
# or in part" but not changing it: each fills its frame (trimmed at the edges, never stretched), is never recoloured or
# retouched, and is credited.
APH_IDS = json.loads(Path(__file__).with_name("aph_ids.json").read_text(encoding="utf-8"))
PHOTO_URL = "https://www.aph.gov.au/api/parliamentarian/{}/image"
PHOTO_CREDIT = "MP photos: Parliament of Australia (CC BY-NC-ND 4.0)."

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,400&display=swap');

/* "The Record": paper, ink and a highlighter. Party colours live in config.BLOC_COLOURS and are never used for the interface. */
:root {
  --paper:#f7f5f0; --paper-2:#efebe2; --card:#fffdf8; --ink:#16140f; --text:#2a2620; --muted:#6b665c; --line:#d9d4c7;
  --hl:#ffe45c; --hl-soft:#fff1a8; --serif:'Newsreader', Georgia, serif; --sans:'Inter', system-ui, sans-serif;
}
html, body, .stApp, .stMarkdown, button, input, textarea, select {font-family:var(--sans);}
.stApp {background:var(--paper); color:var(--text);}
::selection {background:var(--hl);}

/* hide Streamlit's own chrome so the page reads as one product */
[data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer {display:none !important;}
header[data-testid="stHeader"] {background:transparent; height:0;}
/* the page fills a laptop screen (up to a broadsheet's width on a big monitor); long text keeps a column's width, below */
.block-container {max-width:1600px; padding:1.6rem clamp(1rem, 3.5vw, 3.5rem) 3rem;}
/* about 75 characters a line for the record's own words (serif); notes and captions (sans, narrower letters) get a little more */
:is(.quote, .passage, .answer, .then p, .ai-box) {max-width:70ch;}
:is(.notice, .hint, .glance .s, .glance .fine), div[data-testid="stAlert"], [data-testid="stCaptionContainer"],
[data-testid="stMarkdownContainer"] > :is(p, ul, ol, blockquote) {max-width:88ch;}
h1, h2, h3, h4 {color:var(--ink); letter-spacing:-0.01em;}
a, [data-testid="stMarkdownContainer"] a {color:var(--ink) !important; text-decoration:underline; text-decoration-color:#e0c200;
  text-decoration-thickness:2px; text-underline-offset:3px;}

/* masthead and headline */
.mast {display:flex; align-items:baseline; gap:1rem; border-bottom:3px double var(--ink); padding-bottom:.55rem;}
.wordmark {font-family:var(--serif); font-weight:600; font-size:1.65rem; color:var(--ink); letter-spacing:-0.02em; line-height:1;}
.badge {margin-left:auto; font-size:.78rem; font-weight:500; letter-spacing:.02em; color:var(--muted);}
.dateline {display:flex; justify-content:space-between; gap:1rem; font-size:.7rem; letter-spacing:.09em; text-transform:uppercase;
           color:var(--muted); border-bottom:1px solid var(--ink); padding:.35rem 0;}
.hero {padding:0 0 1.2rem 0;}
.hero h1 {font-family:var(--serif); font-weight:500; font-size:3.1rem; line-height:1.06; letter-spacing:-0.025em; margin:1.6rem 0 .8rem 0;
          padding:0;}
[data-testid="stHeaderActionElements"] {display:none;}
.hero p {color:var(--muted); font-size:1.02rem; line-height:1.6; margin:0; max-width:620px;}
.hero.compact {padding-bottom:.6rem;}
.hero.compact h1 {font-size:2rem; margin:1rem 0 .2rem 0;}
mark, .hl {background:linear-gradient(100deg, transparent .5%, var(--hl) 2%, var(--hl) 97%, transparent 99%); color:inherit;
           padding:0 .12em; -webkit-box-decoration-break:clone; box-decoration-break:clone;}

/* every horizontal radio becomes a row of newspaper-style section tabs with an ink underline on the chosen one.
   Streamlit up to ~1.4x draws options as label[data-baseweb="radio"]; newer versions as label[data-testid="stRadioOption"]. */
div[data-testid="stElementContainer"]:has(> div[data-testid="stRadio"]), div[data-testid="stRadio"], div[data-testid="stRadio"] > div {
  width:100% !important;}
div[data-testid="stRadio"] div[role="radiogroup"] {display:flex; flex-wrap:wrap; gap:0 1.6rem; background:transparent; padding:0;
  border-radius:0; border-bottom:1px solid var(--line); width:100%;}
div[data-testid="stRadio"] :is(label[data-baseweb="radio"], label[data-testid="stRadioOption"]) {margin:0 0 -1px 0 !important;
  padding:.6rem 0 .65rem 0; border-radius:0; cursor:pointer; background:transparent !important;}
div[data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child,
div[data-testid="stRadio"] label[data-testid="stRadioOption"] > div > div:not([data-testid="stMarkdownContainer"]) {display:none;}
div[data-testid="stRadio"] :is(label[data-baseweb="radio"], label[data-testid="stRadioOption"]) p {font-size:.92rem; font-weight:500;
  color:var(--muted); margin:0;}
div[data-testid="stRadio"] :is(label[data-baseweb="radio"], label[data-testid="stRadioOption"]):hover p {color:var(--ink);}
div[data-testid="stRadio"] :is(label[data-baseweb="radio"], label[data-testid="stRadioOption"]):has(input:checked) {
  box-shadow:inset 0 -2px 0 var(--ink);}
div[data-testid="stRadio"] :is(label[data-baseweb="radio"], label[data-testid="stRadioOption"]):has(input:checked) p {color:var(--ink);
  font-weight:600;}
.st-key-mode div[data-testid="stRadio"] div[role="radiogroup"] {border-bottom:1px solid var(--ink);}
.st-key-mode :is(label[data-baseweb="radio"], label[data-testid="stRadioOption"]) p {font-size:1rem;}
/* the side-of-politics filter is a set of chips, each with its party colour, rather than another row of tabs */
.st-key-trend_bloc div[data-testid="stRadio"] div[role="radiogroup"] {border-bottom:0; gap:.45rem;}
.st-key-trend_bloc div[data-testid="stRadio"] :is(label[data-baseweb="radio"], label[data-testid="stRadioOption"]) {border:1px solid var(--ink);
  padding:.28rem .8rem; margin:0 !important; box-shadow:none !important;}
.st-key-trend_bloc div[data-testid="stRadio"] :is(label[data-baseweb="radio"], label[data-testid="stRadioOption"]):has(input:checked) {background:var(--ink) !important;}
.st-key-trend_bloc div[data-testid="stRadio"] :is(label[data-baseweb="radio"], label[data-testid="stRadioOption"]):has(input:checked) p {color:var(--paper);}
.st-key-trend_bloc div[data-testid="stRadio"] div[role="radiogroup"] > :nth-child(n+2) p::before {content:"";
  display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:.45rem; background:var(--other); vertical-align:1px;}
.st-key-trend_bloc div[data-testid="stRadio"] div[role="radiogroup"] > :nth-child(2) p::before {background:var(--labor);}
.st-key-trend_bloc div[data-testid="stRadio"] div[role="radiogroup"] > :nth-child(3) p::before {background:var(--coalition);}

/* panels: a rule across the top instead of a rounded box (bordered containers; only those draw a border) */
div[data-testid="stVerticalBlockBorderWrapper"], div[data-testid="stVerticalBlock"] {border-radius:0 !important;
  border-color:var(--line) !important; border-top-color:var(--ink) !important; border-top-width:2px !important;}
.hint {color:var(--muted); font-size:.88rem; margin:.15rem 0 .2rem 0;}

/* inputs: white with an ink edge and a highlighter focus ring */
div[data-testid="stTextInputRootElement"], div[data-testid="stTextInput"] div[data-baseweb="input"],
div[data-testid="stSelectbox"] div[role="group"], div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
  background:var(--card) !important; border:1.5px solid var(--ink) !important; border-radius:2px !important;}
div[data-testid="stTextInputRootElement"]:focus-within, div[data-testid="stTextInput"] div[data-baseweb="input"]:focus-within,
div[data-testid="stSelectbox"] div[role="group"]:focus-within, div[data-testid="stSelectbox"] div[data-baseweb="select"] > div:focus-within {
  box-shadow:0 0 0 3px var(--hl) !important;}
div[data-testid="stTextInput"] input, div[data-testid="stSelectbox"] input {font-size:1rem; color:var(--ink);}
div[data-testid="stWidgetLabel"] p {font-weight:500; color:var(--ink); font-size:.88rem;}

/* buttons: square, outlined in ink, highlighted on hover */
.stButton button, .stDownloadButton button {border-radius:2px; border:1px solid var(--ink); background:transparent; color:var(--ink);
  font-weight:500; padding:.35rem 1rem; transition:background .12s;}
.stButton button:hover, .stDownloadButton button:hover, .stButton button:focus-visible {background:var(--hl); border-color:var(--ink);
  color:var(--ink);}
.stButton button:active {background:var(--hl-soft);}

/* three-step explainer */
.steps {display:grid; grid-template-columns:repeat(3, 1fr); gap:28px; margin:2rem 0 1.4rem 0;}
.step {border-top:1px solid var(--ink); padding:.7rem 0 0 0;}
.step .n {font-size:.7rem; font-weight:600; color:var(--muted); letter-spacing:.09em; text-transform:uppercase;}
.step b {display:block; color:var(--ink); margin:.3rem 0 .25rem 0; font-family:var(--serif); font-weight:500; font-size:1.35rem;
         line-height:1.2;}
.step span.t {color:var(--muted); font-size:.92rem; line-height:1.5;}
@media (max-width: 700px) {.steps {grid-template-columns:1fr; gap:14px;} .hero h1 {font-size:2.15rem;} .dateline span + span {display:none;}}

/* MP header */
.mp {display:flex; gap:1rem; align-items:center; padding:1.2rem 0 1rem 0; border-bottom:1px solid var(--ink); margin:.4rem 0 .6rem 0;}
.mp .avatar {flex:none; position:relative; width:64px; height:92px; background:var(--paper-2); color:var(--ink);
             border:1px solid var(--ink); display:grid; place-items:center; font-family:var(--serif); font-weight:500; font-size:1.45rem;}
/* the official portrait is laid over the initials and fills the frame (the frame has the portraits' usual shape, so little is
   trimmed, and the trim keeps the face: more off the bottom than the top); if it cannot load, the initials show */
.mp .avatar .photo {position:absolute; inset:0; background-position:center 20%; background-size:cover; background-repeat:no-repeat;}
.mp .kicker {font-size:.7rem; font-weight:600; letter-spacing:.09em; text-transform:uppercase; color:var(--muted);}
.mp .name {font-family:var(--serif); font-size:2.1rem; font-weight:500; color:var(--ink); line-height:1.1; letter-spacing:-0.02em;}
.mp .meta {color:var(--muted); font-size:.9rem; margin-top:.35rem; display:flex; flex-wrap:wrap; gap:.3rem 1rem; align-items:center;}
.party {display:inline-flex; align-items:center; gap:.4rem; font-weight:600; color:var(--ink);}
.party i {display:inline-block; width:10px; height:10px; border-radius:50%; background:var(--dot);}
.chip {display:inline-block; font-size:.75rem; font-weight:600; padding:.15rem .6rem; background:var(--paper-2); color:var(--ink);}
.notice {background:#fff6cc; border-left:3px solid #d9b100; color:#4d3d00; padding:.6rem .9rem; font-size:.88rem; margin:.4rem 0;}

/* section titles: a stroke of highlighter, then a serif heading */
.sec {margin:2.3rem 0 .7rem 0;}
.sec::before {content:""; display:block; width:34px; height:5px; background:var(--hl); margin-bottom:.55rem;}
.sec h4 {margin:0; padding:0; font-family:var(--serif); font-size:1.5rem; font-weight:500; letter-spacing:-0.015em;}
.sec p {margin:.25rem 0 0 0; color:var(--muted); font-size:.9rem; max-width:720px;}

/* headline numbers */
.stats {display:grid; grid-template-columns:repeat(4, 1fr); gap:24px; margin:1.1rem 0 .4rem 0;}
.stat {border-top:2px solid var(--ink); padding:.55rem 0 0 0;}
.stat .l {color:var(--muted); font-size:.7rem; font-weight:600; letter-spacing:.08em; text-transform:uppercase;}
.stat .v {color:var(--ink); font-family:var(--serif); font-size:2.2rem; font-weight:500; margin-top:.1rem; line-height:1.15;
          font-variant-numeric:lining-nums;}
.stat .v.small {font-size:1.25rem; line-height:1.3; margin-top:.35rem;}
@media (max-width: 700px) {.stats {grid-template-columns:repeat(2, 1fr); gap:16px;}}
div[data-testid="stMetric"] {background:transparent; border:0; border-top:2px solid var(--ink); border-radius:0; padding:.55rem 0 0 0;}
div[data-testid="stMetricValue"] {font-family:var(--serif); font-weight:500;}
[data-testid="stMetricLabel"] p {color:var(--muted);}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {white-space:normal !important; overflow:visible !important;
  text-overflow:clip !important;}

/* quotes from the record; the highlighter marks only the words that matter (the search terms, or a promise and its number) */
.quote {border-left:3px solid var(--ink); padding:.15rem 0 .15rem 1.15rem; margin:1.1rem 0; color:var(--ink);
        font-family:var(--serif); font-size:1.2rem; line-height:1.55;}
:is(.quote, .passage, .then) mark {background:linear-gradient(transparent 52%, var(--hl) 52%, var(--hl) 94%, transparent 94%);
  padding:0 .04em; -webkit-box-decoration-break:clone; box-decoration-break:clone;}
.quote .cite {font-family:var(--sans); margin-top:.45rem;}
.cite {color:var(--muted); font-size:.8rem; line-height:1.5;}
.cite a {color:var(--muted); text-underline-offset:2px;}
.cite a:hover {color:var(--ink); background:var(--hl-soft);}
.quote .cite b {color:var(--ink); font-weight:600;}

/* "at a glance": three questions about the MP, each answered in plain words with its evidence one click away */
.glance {border-top:2px solid var(--ink); padding-top:.65rem;}
.glance .q {font-family:var(--serif); font-size:1.3rem; font-weight:500; line-height:1.2; color:var(--ink);}
.glance .a {font-family:var(--serif); font-size:2.7rem; font-weight:500; line-height:1; color:var(--ink); margin:.6rem 0 .3rem 0;
            font-variant-numeric:lining-nums;}
.glance .a.words {font-size:1.45rem; line-height:1.25;}
.glance .s {color:var(--text); font-size:.95rem; line-height:1.5;}
.glance .fine {color:var(--muted); font-size:.8rem; line-height:1.45; margin-top:.4rem;}
.glance .quote {font-size:1.02rem; margin:.7rem 0 .2rem 0;}
.bars {display:flex; flex-direction:column; gap:.45rem; margin:.7rem 0 .2rem 0;}
.bars .row {display:grid; grid-template-columns:minmax(0, 1fr) 2.6rem; gap:.2rem .6rem; align-items:center; font-size:.92rem;}
.bars .row b {font-weight:500; color:var(--ink);}
.bars .row span {text-align:right; color:var(--muted); font-variant-numeric:tabular-nums;}
.bars .row i {grid-column:1 / -1; display:block; height:6px; background:var(--paper-2);}
.bars .row i::after {content:""; display:block; height:100%; width:var(--w); background:var(--ink);}

/* a Question Time answer shaded sentence by sentence: the more directly a sentence responds to the question, the stronger the
   highlighter (scored by the same cross-encoder the directness estimate uses) */
.answer {font-family:var(--serif); font-size:1.02rem; line-height:1.75; color:var(--text);}
.answer span {padding:.08em 0; -webkit-box-decoration-break:clone; box-decoration-break:clone;}
.answer .r3 {background:var(--hl);} .answer .r2 {background:var(--hl-soft);} .answer .r1 {background:#fff8da;}
.legend {display:flex; flex-wrap:wrap; gap:.3rem 1.1rem; align-items:center; font-size:.75rem; color:var(--muted); margin:.5rem 0 .3rem 0;}
.legend i {display:inline-block; width:15px; height:10px; margin-right:.35rem; vertical-align:-1px; border:1px solid var(--line);}

/* one MP's share on a 0-100% scale: the 95% range as a band, the estimate as a dot, the average as a dashed line */
.rangebar {margin:.9rem 0 1rem 0; max-width:420px;}
.rangebar .track {position:relative; height:24px; border-bottom:1px solid var(--ink);}
.rangebar .band {position:absolute; top:10px; height:5px; background:var(--ink); opacity:.28;}
.rangebar .avg {position:absolute; top:0; bottom:-5px; border-left:1.5px dashed var(--muted);}
.rangebar .dot {position:absolute; top:7px; width:11px; height:11px; margin-left:-5.5px; border-radius:50%; background:var(--ink);}
.rangebar .scale {position:relative; height:1rem; font-size:.72rem; color:var(--muted); margin-top:.3rem;}
.rangebar .scale span {position:absolute; transform:translateX(-50%);}
.rangebar .scale span:first-child {transform:none;} .rangebar .scale span:last-child {transform:translateX(-100%);}
.rangebar .key {font-size:.75rem; color:var(--muted); margin-top:.15rem;}

/* the postcode panel and the seat map sit side by side on wide screens and stack below 900px (the map is the row's iframe).
   Side by side, the map grows with the window: on the empty screen it fills the first screenful, and once an MP is chosen it
   stays short enough for their record to start above the fold. */
@media (min-width: 901px) {
  .st-key-seat_map_find iframe {height:clamp(440px, calc(100vh - 390px), 620px);}
  .stApp:has(.mp) .st-key-seat_map_find iframe {height:clamp(330px, calc(100vh - 540px), 420px);}
  .st-key-seat_map_compare iframe {height:clamp(420px, calc(100vh - 380px), 560px);}  /* Compare MPs: the map spans the page */
}
@media (max-width: 900px) {
  /* once an MP is chosen, the stacked map becomes a short strip so their record starts sooner on a phone */
  .stApp:has(.mp) div[data-testid="stHorizontalBlock"]:has(> div[data-testid="stColumn"] iframe) iframe {height:240px !important;}
  div[data-testid="stHorizontalBlock"]:has(> div[data-testid="stColumn"] iframe) {flex-direction:column; gap:1rem;}
  div[data-testid="stHorizontalBlock"]:has(> div[data-testid="stColumn"] iframe) > div[data-testid="stColumn"] {width:100% !important;
    flex:1 1 auto !important; min-width:100% !important;}
}

/* earliest vs latest wording: two dated newspaper columns */
.then {border-top:2px solid var(--ink); padding-top:.5rem;}
.then .yr {font-family:var(--serif); font-size:2rem; font-weight:500; color:var(--ink); line-height:1.1;}
.then .yr span {display:block; font-family:var(--sans); font-size:.7rem; font-weight:600; letter-spacing:.09em; text-transform:uppercase;
                color:var(--muted);}
.then p {font-family:var(--serif); font-size:1.02rem; line-height:1.6; color:var(--text); margin:.5rem 0 .4rem 0;}

/* who spoke */
.who {display:flex; align-items:center; gap:.8rem; padding:.65rem 0; border-bottom:1px solid var(--line); max-width:52rem;}
.who .nm {font-weight:600; color:var(--ink); min-width:0;}
.who .sub {color:var(--muted); font-size:.85rem; display:flex; align-items:center; gap:.4rem;}
.who .sub i {display:inline-block; width:8px; height:8px; border-radius:50%; background:var(--dot);}
.who .bar {margin-left:auto; flex:none; width:160px; height:8px; background:var(--paper-2);}
.who .bar i {display:block; height:100%; background:var(--ink);}
.who .cnt {flex:none; width:70px; text-align:right; color:var(--muted); font-size:.82rem; font-variant-numeric:tabular-nums;}
@media (max-width: 700px) {.who .bar {width:70px;}}

/* expanders, alerts, AI answer, footer */
div[data-testid="stExpander"] details {border-radius:0; border:0; border-top:1px solid var(--line); background:transparent;}
div[data-testid="stExpander"] details summary p {font-weight:500; color:var(--ink);}
div[data-testid="stExpander"] details summary:hover p {background:var(--hl-soft);}
/* messages ("no clear match", a bad postcode, ...): paper and ink with a coloured rule, not Streamlit's blue, yellow and red boxes */
div[data-testid="stAlert"] > div {border-radius:0 !important; background:var(--paper-2) !important; color:var(--ink) !important;
  border-left:3px solid var(--ink) !important;}
div[data-testid="stAlert"] p {color:inherit !important;}
div[data-testid="stAlert"]:has([data-testid="stAlertContentWarning"]) > div {background:#fff6cc !important; color:#4d3d00 !important;
  border-left-color:#d9b100 !important;}
div[data-testid="stAlert"]:has([data-testid="stAlertContentError"]) > div {background:#f8e9e4 !important; color:#5c2116 !important;
  border-left-color:#b5523b !important;}
.passage {font-family:var(--serif); font-size:1.05rem; line-height:1.65; color:var(--text); margin:.2rem 0 .5rem 0;}
.ai-box {background:var(--hl-soft); border-left:3px solid var(--ink); padding:.7rem 1rem; color:var(--text); margin:.5rem 0 .4rem 0;}
.foot {color:var(--muted); font-size:.78rem; margin-top:3.5rem; border-top:3px double var(--ink); padding-top:.9rem; line-height:1.6;
       padding-right:max(0px, calc(100% - 130ch));}  /* the rule runs the full width; the fine print wraps sooner */
.foot b {color:var(--ink);}
</style>
"""


def apply_style():
    """Inject the app's stylesheet, plus the side-of-politics colours as CSS variables (--labor, --coalition, --other)."""
    party_vars = "".join(f"--{side.lower()}:{colour};" for side, colour in BLOC_COLOURS.items())
    st.markdown(CSS + f"<style>:root {{{party_vars}}}</style>", unsafe_allow_html=True)


def hero(badge: str, compact: bool = False):
    """Page header: masthead, one-line promise, and a small status badge (AI answers on or off).

    `compact` (once someone has been chosen) keeps the headline small and drops the introduction, so the record starts sooner.
    """
    intro = (
        ""
        if compact
        else "<p>An MP (Member of Parliament) is the person who represents your local area. Everything here comes from Hansard, "
        "the official written record of the House of Representatives, March 1998 to July 2025, with the source for every quote.</p>"
    )
    st.markdown(
        f"<div class='hero{' compact' if compact else ''}'><div class='mast'><span class='wordmark'>Parlianote</span>"
        f"<span class='badge'>{esc(badge)}</span></div>"
        "<div class='dateline'><span>House of Representatives &middot; Hansard</span><span>On the record since March 1998</span></div>"
        f"<h1>What your MP <mark>actually says</mark> in Parliament.</h1>{intro}</div>",
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


def party_dot(abbrev: str) -> str:
    """Inline style giving a party marker the colour of its side of politics (Labor, Coalition or Other)."""
    return f"--dot:{BLOC_COLOURS[bloc(abbrev)]}"


def photo_html(mp_key: str, name: str) -> str:
    """The MP's official portrait as a layer for their initials box, or "" if we have no ID for them (the initials stay)."""
    photo_id = APH_IDS.get(str(mp_key))
    if not photo_id:
        return ""
    return (
        f"<span class='photo' role='img' aria-label='Photo of {esc(name)}' title='Photo: Parliament of Australia' "
        f"style=\"background-image:url('{PHOTO_URL.format(esc(photo_id))}')\"></span>"
    )


def mp_header(info):
    """The chosen MP: official photo (initials until it loads, or if it cannot), name, party, area and how much of their record we hold."""
    name = display_name(str(info["name"]))
    initials = "".join(part[0] for part in name.split()[:1] + name.split()[-1:]).upper()
    st.markdown(
        f"<div class='mp'><div class='avatar'>{esc(initials)}{photo_html(info['mp_key'], name)}</div><div>"
        f"<div class='kicker'>Member of Parliament</div><div class='name'>{esc(name)}</div>"
        f"<div class='meta'><span class='party' style='{party_dot(info['party'])}'><i></i>{esc(party_label(info['party']))}</span>"
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
        f"<div class='sub'><i style='{party_dot(r.party)}'></i>{esc(party_label(r.party))} &middot; {esc(r.electorate)}</div></div>"
        f"<div class='bar'><i style='width:{100 * r.passages / top:.0f}%'></i></div>"
        f"<div class='cnt'>{r.passages} of {total}</div></div>"
        for r in who.itertuples()
    )
    st.markdown(rows, unsafe_allow_html=True)


def footer(extra: str = ""):
    """Sources and disclaimer shown at the bottom of every screen."""
    st.markdown(
        "<div class='foot'><b>Always check quotes against the official record.</b><br>Source: Hansard, House of Representatives, March 1998 to July 2025, via Katz &amp; Alexander (2025), CC BY 4.0. "
        f"Postcodes: Matthew Proctor. Electoral boundaries: Australian Electoral Commission (2025 election). {PHOTO_CREDIT} "
        "Parlianote is a student project and is not affiliated with the Parliament of Australia. "
        f"{extra}</div>",
        unsafe_allow_html=True,
    )


# Words that never get a highlight, even when they were typed: "what has she said about housing" marks only "housing".
_UNMARKED = {
    "about", "after", "also", "been", "being", "does", "from", "have", "into", "just", "more", "most", "much", "only", "over", "said",
    "says", "some", "than", "that", "their", "them", "then", "there", "they", "this", "those", "very", "were", "what", "when", "where",
    "which", "while", "will", "with", "would", "your", "and", "are", "but", "for", "has", "her", "his", "its", "not", "our", "she", "the",
    "was", "who", "why", "how", "you", "any", "all", "can", "did", "had", "him", "one", "out", "two",
}


def query_marks(query: str):
    """A pattern for the searched words (and their endings: "gambling" also marks "gamble"), or None if nothing is worth marking."""
    stems = []
    for word in re.findall(r"[a-z]{3,}", str(query).lower()):
        if word in _UNMARKED:
            continue
        stem = re.sub(r"(ing|ies|es|ed|s)$", "", word) if len(word) > 5 else word
        stems.append(re.escape(stem))
    return re.compile(r"\b(?:" + "|".join(sorted(set(stems), key=len, reverse=True)) + r")\w*", re.I) if stems else None


def marked(text: str, marks=None) -> str:
    """Escaped text, with every match of `marks` (one compiled pattern) wrapped in a highlighter <mark>."""
    text = html.escape(" ".join(str(text).split()), quote=False)
    return marks.sub(lambda m: f"<mark>{m.group(0)}</mark>", text) if marks else text


def passage(text: str, marks=None):
    """A passage from the record, shown as plain text: escaped HTML, so '$30 ... $100' is not read as a maths formula."""
    st.markdown(f"<div class='passage'>{marked(text, marks)}</div>", unsafe_allow_html=True)


def quote_card(text: str, source_html: str, marks=None):
    """A quoted sentence with its source underneath; `marks` highlights the words that matter. `source_html` must already be escaped."""
    st.markdown(f"<div class='quote'>&ldquo;{marked(text, marks)}&rdquo;<div class='cite'>{source_html}</div></div>", unsafe_allow_html=True)


def glance(question: str, answer: str, sentence: str = "", fine: str = "", words: bool = False):
    """One "at a glance" card: the question, a big answer (a number, or a few words), a plain sentence and the small print.

    All arguments are HTML; text taken from the data must be escaped by the caller.
    """
    st.markdown(
        f"<div class='glance'><div class='q'>{question}</div><div class='a{' words' if words else ''}'>{answer}</div>"
        f"<div class='s'>{sentence}</div>{f'<div class=fine>{fine}</div>' if fine else ''}</div>",
        unsafe_allow_html=True,
    )


def range_bar(share: float, low: float, high: float, average: float, who: str = "this MP"):
    """A share on a 0-100% scale with its 95% range, against the average (a dashed line)."""
    pc = lambda x: f"{100 * min(max(x, 0), 1):.1f}%"  # noqa: E731
    st.markdown(
        f"<div class='rangebar' role='img' aria-label='{esc(who)}: {share:.0%}, 95% range {low:.0%} to {high:.0%}; average {average:.0%}'>"
        f"<div class='track'><i class='band' style='left:{pc(low)};width:{pc(high - low)}'></i>"
        f"<i class='avg' style='left:{pc(average)}'></i><i class='dot' style='left:{pc(share)}'></i></div>"
        "<div class='scale'><span style='left:0'>0%</span><span style='left:50%'>50%</span><span style='left:100%'>100%</span></div>"
        f"<div class='key'>&#9679; {esc(who)}, with its 95% range &nbsp;&middot;&nbsp; dashed line: average, {average:.0%}</div></div>",
        unsafe_allow_html=True,
    )


RELEVANCE_CUTS = (-4.5, -3.0, -1.0)  # cross-encoder scores: barely / partly / clearly responds (set on real direct and indirect answers)


def shaded_answer(scored):
    """An answer with each sentence shaded by how directly it responds to the question. `scored` is [(sentence, score), ...]."""
    levels = [sum(x >= c for c in RELEVANCE_CUTS) for _, x in scored]
    first = next((i for i, level in enumerate(levels) if level >= 2), None)
    summary = (
        f"The answer first responds to the question in <b>sentence {first + 1} of {len(scored)}</b>."
        if first is not None
        else f"None of its {len(scored)} sentences clearly responds to the question."
    )
    spans = " ".join(f"<span class='r{level}'>{esc(text)}</span>" for (text, _), level in zip(scored, levels))
    st.markdown(
        f"<div class='hint'>{summary}</div><div class='answer'>{spans}</div>"
        "<div class='legend'><span><i style='background:var(--hl)'></i>responds to the question</span>"
        "<span><i style='background:var(--hl-soft)'></i>partly</span><span><i style='background:#fff8da'></i>barely</span>"
        "<span><i></i>not at all</span></div>",
        unsafe_allow_html=True,
    )


def share_bars(rows):
    """Small labelled bars: rows of (label, share between 0 and 1)."""
    cells = "".join(
        f"<div class='row'><b>{esc(label)}</b><span>{share:.0%}</span><i style='--w:{100 * share:.0f}%'></i></div>" for label, share in rows
    )
    st.markdown(f"<div class='bars'>{cells}</div>", unsafe_allow_html=True)


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


def example_buttons(label: str, items, key_prefix: str, target_key: str, fmt=lambda item: item, per_row: int | None = None):
    """Rows of buttons (all in one row unless `per_row` is given); clicking one fills the input widget stored under `target_key`."""
    st.caption(label)
    per_row = per_row or len(items)
    for start in range(0, len(items), per_row):
        for col, item in zip(st.columns(per_row), items[start : start + per_row]):
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
