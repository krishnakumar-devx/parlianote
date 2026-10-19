"""Parlianote: see what your MP actually says in Parliament, with the source for every quote.

Run from the project root:   streamlit run app/app.py
Optional AI answers:         put a key in .env (see .env.example and the README).

This file only decides which screen to show. The screens are in src/ui/views.py, the data loading in src/ui/data.py,
the search helpers in src/ui/search.py and the reusable pieces in src/ui/components.py.
"""

import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so `src` can be imported when run through streamlit

from src.ui import components as C  # noqa: E402
from src.ui import views  # noqa: E402
from src.ui.config import ELECTION_2025, EXAMPLE_POSTCODES, MODES, VIEWS  # noqa: E402
from src.ui.data import ai_provider, load_lookup, load_mp_list, load_speeches, warm_up  # noqa: E402
from src.ui.seatmap import seat_map  # noqa: E402

st.set_page_config(page_title="Parlianote", page_icon="🏛️", layout="wide", initial_sidebar_state="collapsed")
C.apply_style()
warm_up()  # start loading the language models in the background now, so the first search does not wait for them

postcodes, latest_speaker = load_lookup()
mps = load_mp_list()
provider = ai_provider()


# ------------------------------------------------------------------------------------------------ remember choices
# Links such as ?mode=Search%20all%20MPs&q=housing preselect a screen; the widgets below keep their values in session_state.
query = st.query_params
mp_names = ["(use postcode)"] + mps["name"].tolist()
start_mode = query.get("mode") if query.get("mode") in MODES else MODES[0]
st.session_state.setdefault("mode", start_mode)
st.session_state.setdefault("postcode", query.get("postcode", ""))
st.session_state.setdefault("global_q", query.get("q", "") if start_mode == MODES[1] else "")
st.session_state.setdefault("mp_q", query.get("q", "") if start_mode == MODES[0] else "")
st.session_state.setdefault("mp_pick", query.get("mp") if query.get("mp") in mp_names else mp_names[0])
st.session_state.setdefault("view", query.get("view") if query.get("view") in VIEWS else VIEWS[0])
st.session_state.setdefault("cmp_a", query.get("a") if query.get("a") in mp_names else "(choose an MP)")
st.session_state.setdefault("cmp_b", query.get("b") if query.get("b") in mp_names else "(choose an MP)")
st.session_state.setdefault("kiosk", query.get("kiosk") is not None)  # ?kiosk=1: showcase mode (quick idle tour, resets itself)


# ------------------------------------------------------------------------------------------------ header and mode
# Once someone has been chosen (or searched for), the header shrinks so their record starts higher up the page.
mode, state = st.session_state["mode"], st.session_state
chosen = (
    (mode == MODES[0] and bool(state.get("postcode") or state.get("mp_pick") != mp_names[0]))
    or (mode == MODES[1] and bool(state.get("global_q")))
    or (mode == MODES[2] and "(choose an MP)" not in (state.get("cmp_a"), state.get("cmp_b")))
)
C.hero(f"AI answers on · {provider['name']}" if provider else f"{len(mps)} MPs · 1998–2025", compact=chosen)
st.radio(
    "What would you like to do?", MODES, horizontal=True, key="mode", label_visibility="collapsed", on_change=st.query_params.clear
)  # a new screen starts with a clean address bar; each screen then writes its own shareable link
footnote = (
    f"AI answers: {provider['name']} ({provider['model']})." if provider else "AI answers are off; add a key to .env to turn them on."
)

if st.session_state["mode"] != MODES[0]:
    {MODES[1]: views.search_all, MODES[2]: lambda: views.compare_mps(mps), MODES[3]: views.parliament_trends}[st.session_state["mode"]]()
    C.footer(footnote)
    st.stop()


# ------------------------------------------------------------------------------------------------ find my MP
def find_mp():
    """Draw the postcode box and MP picker. Returns (mp_key or None, note for under the MP card, postcode typed, name picked)."""
    with st.container(border=True):
        col_postcode, col_name = st.columns([1, 2], gap="large")
        with col_postcode:
            postcode = st.text_input("Your postcode", max_chars=4, placeholder="e.g. 2088", key="postcode").strip()
        with col_name:
            picked = st.selectbox("Or choose an MP by name", mp_names, key="mp_pick")

    if picked != "(use postcode)":
        return mps.loc[mps["name"] == picked, "mp_key"].iloc[0], None, postcode, picked
    if not postcode:
        return None, None, postcode, picked
    if not re.fullmatch(r"\d{4}", postcode):
        st.error("Please enter a 4-digit Australian postcode, for example 2088.")
        return None, None, postcode, picked
    if re.match(r"^(1|9|02)", postcode):  # PO-box and business-only ranges are not home addresses
        st.warning(
            "That postcode is used for PO boxes or large businesses, not home addresses, so it is not linked to a local area. "
            "Please use your home postcode, or pick an MP by name."
        )
        return None, None, postcode, picked

    rows = postcodes[postcodes["postcode"] == postcode]
    if rows.empty:
        st.warning("We could not match that postcode to a local area (electorate). Check the number, or pick an MP by name instead.")
        return None, None, postcode, picked

    electorates = sorted(rows["electorate"].unique())
    electorate, note = electorates[0], None
    if len(electorates) > 1:
        electorate = st.selectbox("This postcode covers more than one local area (electorate). Choose yours:", electorates)
        note = "Some postcodes cover more than one area. If you are unsure which is yours, check on the Australian Electoral Commission (AEC) website."
    latest = latest_speaker[latest_speaker["electorate"] == electorate]
    if latest.empty:
        st.warning(f"We have no recorded speeches for {electorate}.")
        return None, None, postcode, picked
    last_seen = pd.Timestamp(latest["last_seen"].iloc[0])
    if last_seen < ELECTION_2025:
        note = (
            f"The most recent speaker we have for {electorate} last spoke on {last_seen:%d %b %Y}, before the 3 May 2025 federal election, "
            "so this area may now have a different MP. Pick an MP by name or check with the AEC."
        )
    return latest["mp_key"].iloc[0], note, postcode, picked


# Postcode box and the chosen MP on the left; the map of their seat on the right (stacked on a phone).
col_find, col_map = st.columns([7, 5], gap="large")
with col_find:
    mp_key, note, postcode, picked = find_mp()
    info = mps[mps["mp_key"] == mp_key].iloc[0] if mp_key is not None else None
    if info is not None:
        C.mp_header(info)
        if note:
            C.notice(note)
    elif not postcode:
        C.example_buttons(
            "Try an example postcode:", EXAMPLE_POSTCODES, "ex_pc", "postcode", fmt=lambda item: f"{item[0]} ({item[1]})", per_row=3
        )
        C.how_it_works()
with col_map:  # full size on the empty screen; smaller once an MP is chosen, so their record starts above the fold
    seat_map(info["electorate"] if info is not None else None, height=330 if info is not None else 440, tour=True)

if mp_key is None:
    if not postcode:
        views.on_this_day()
    C.footer(footnote)
    st.stop()

speeches = load_speeches(mp_key)
st.query_params.update(
    {"mode": MODES[0], "postcode": postcode} if postcode and picked == "(use postcode)" else {"mode": MODES[0], "mp": picked}
)

st.radio("View", VIEWS, horizontal=True, label_visibility="collapsed", key="view")
view = st.session_state["view"]
if view == "At a glance":
    views.at_a_glance(mp_key, info, speeches)
elif view == "Overview":
    views.overview(speeches)
elif view == "Ask the record":
    views.ask_the_record(mp_key, info, speeches)
elif view == "Question Time":
    views.question_time(mp_key)
elif view == "Promises":
    views.promises(mp_key)
else:
    views.about()
C.footer(footnote)
