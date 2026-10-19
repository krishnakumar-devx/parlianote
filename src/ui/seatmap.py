"""The seat map on "Find my MP" (and, for two MPs at once, "Compare MPs"): the MP's state lifts off the page, then their seat rises out of it.

Shapes are in src/ui/seat_map/seat_map.json (AEC boundaries used at the 2025 election, built by scripts/build_seat_map.py). The map
is a small Streamlit component (src/ui/seat_map/index.html) that loads those shapes once and stays on the page: each time the chosen
seat changes, Python sends it a short "scene" (which states lift, which seats rise, the caption, by name) and the map animates there
from wherever it is. On the empty "Find my MP" screen it also gets a tour, which it plays when nobody has touched the page for a while.
"""

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.blocs import bloc
from src.commitment import split_sentences
from src.topics import TOPIC_NAMES
from src.ui.config import ELECTION_2025, PARTY_NAMES
from src.ui import quotes
from src.ui.data import cite, load_lookup, load_mp_list, load_speeches

COMPONENT = Path(__file__).with_name("seat_map")
SEAT_MAP = COMPONENT / "seat_map.json"  # served to the map from the component's own folder, so Python never has to send shapes
_component = components.declare_component("seat_map", path=str(COMPONENT))

# Seats renamed since, as they still appear in the record -> the seat covering much the same area at the 2025 election.
RENAMED = {
    "batman": "Cooper",
    "mcmillan": "Monash",
    "melbourneports": "Macnamara",
    "murray": "Nicholls",
    "denison": "Clark",
    "throsby": "Whitlam",
    "wakefield": "Spence",
    "prospect": "McMahon",
}
# Seats that no longer exist, with their state, so the map can still show roughly where they were.
ABOLISHED = {
    "higgins": "VIC",
    "northsydney": "NSW",
    "stirling": "WA",
    "portadelaide": "SA",
    "charlton": "NSW",
    "kalgoorlie": "WA",
    "lowe": "NSW",
    "gwydir": "NSW",
    "burke": "VIC",
    "bonython": "SA",
    "namadgi": "ACT",
    "northernterritory": "NT",
}
# Seat tops on the "Compare MPs" map: the first MP in highlighter yellow, the second in the ochre of the compare chart's second series.
TOPS = ("#ffe45c", "#c9a227")


def _key(name: str) -> str:
    return re.sub(r"[^a-z]", "", str(name).lower())


@st.cache_data(show_spinner=False)
def _shapes():
    """The boundary file, or None if it has not been built (the map is then simply left out)."""
    return json.loads(SEAT_MAP.read_text(encoding="utf-8")) if SEAT_MAP.exists() else None


def locate(electorate: str | None):
    """Where an electorate from the record is on today's map: (seat key or None, state code or None, note or None)."""
    g = _shapes()
    if g is None or not electorate:
        return None, None, None
    k = _key(electorate)
    if k in g["seats"]:
        return k, g["seats"][k]["state"], None
    if k in RENAMED:
        new = _key(RENAMED[k])
        return new, g["seats"][new]["state"], f"{electorate} is now called {RENAMED[k]}."
    if k in ABOLISHED:
        state = ABOLISHED[k]
        return None, state, f"{electorate} no longer exists, so the map shows {g['states'][state]['name']}."
    return None, None, None


FLAT = {"mode": "flat", "states": [], "seats": [], "close": False, "caption": {"state": "", "seat": "", "name": "", "rows": [], "note": "", "aria": ""}}


def _scene(electorate: str | None) -> dict:
    """One MP's scene: their state lifts, then their seat rises (or the state alone, for a seat that no longer exists)."""
    g = _shapes()
    seat, state, note = locate(electorate)
    if not state:
        return FLAT
    st_name = g["states"][state]["name"]
    name = g["seats"][seat]["name"] if seat else str(electorate)
    return {
        "mode": "single",
        "states": [state],
        "seats": [{"key": seat, "tops": [TOPS[0]]}] if seat else [],
        "close": bool(seat),
        "caption": {
            "state": st_name,
            "seat": name if seat else "",
            "name": name,
            "rows": [],
            "note": note or "",
            "aria": f"Map of Australia showing the seat of {name}, {st_name}" if seat else f"Map of Australia showing {st_name}",
        },
    }


def _draw(scene: dict, height: int, key: str, tour: list | None = None):
    """Send a scene (and, on the empty screen, the tour) to the map component."""
    kiosk = bool(st.session_state.get("kiosk"))
    _component(scene=scene, height=height, tour=tour or [], idle=8 if kiosk else 30, kiosk=kiosk, key=key, default=None)


def seat_map(electorate: str | None, height: int = 440, key: str = "seat_map_find", tour: bool = False):
    """"Find my MP": flat Australia when nothing is chosen, otherwise this electorate's state lifts and then its seat rises.

    With `tour`, the map also gets the idle tour, which it plays only while nothing is chosen and nobody is using the page."""
    if _shapes() is None:
        return
    _draw(_scene(electorate), height, key, tour_stops() if tour else None)


def seat_map_pair(a: tuple[str, str], b: tuple[str, str], height: int = 420, key: str = "seat_map_compare"):
    """The "Compare MPs" map: both MPs' states lift, then both seats rise, the first topped in highlighter yellow, the second in ochre.

    `a` and `b` are (electorate, MP name). Same state: the camera closes in on the two seats. Different states, or a seat that no
    longer exists (shown at state level, as on "Find my MP"): the camera keeps the whole state(s) in view.
    """
    g = _shapes()
    if g is None:
        return
    states, seats, rows, notes = [], [], [], []
    for (electorate, who), top in zip((a, b), TOPS):
        seat, state, note = locate(electorate)
        if state and state not in states:
            states.append(state)
        label = g["seats"][seat]["name"] if seat else str(electorate)
        if seat:  # two members for the same seat share one stack, its top split between them
            same = next((s for s in seats if s["key"] == seat), None)
            if same:
                same["tops"].append(top)
            else:
                seats.append({"key": seat, "tops": [top]})
        rows.append({"label": label, "who": who, "top": top, "gone": not seat})  # gone: only the state is drawn
        notes.append(note or "")
    if rows[0]["label"] == rows[1]["label"]:
        notes.insert(0, "Both have represented the same seat.")
    names = [g["states"][s]["name"] for s in states]
    scene = {
        "mode": "pair",
        "states": states,
        "seats": seats,
        "close": len(states) == 1 and not any(r["gone"] for r in rows),
        "caption": {
            "state": " & ".join(names),
            "seat": "",
            "name": "",
            "rows": rows,
            "note": " ".join(dict.fromkeys(n for n in notes if n)),
            "aria": "Map of Australia comparing "
            + " and ".join(f"the seat of {r['label']} ({r['who']})" for r in rows)
            + (f", {' and '.join(names)}" if names else ""),
        },
    }
    _draw(scene if states else FLAT, height, key)


# --------------------------------------------------------------------------------------------- the idle tour (for a showcase)
TOUR_ORDER = ["NSW", "QLD", "NT", "WA", "SA", "VIC", "TAS", "ACT"]  # a loop around the country
def _tour_quote(mp_key: str, seat: str):
    """One short, self-contained, non-partisan sentence by this MP on their most common subject (not "Partisan debate").

    Sentences about their own seat or community, or with a concrete number, are preferred; among equals, the most recent wins.
    """
    sp = load_speeches(mp_key)
    sp = sp[(sp["topic"] >= 0) & (sp["topic"].map(TOPIC_NAMES) != "Partisan debate")]
    if sp.empty:
        return None
    top = sp["topic"].mode().iloc[0]
    best, best_score = None, -1
    for r in sp[sp["topic"] == top].sort_values("date", ascending=False).head(80).itertuples():
        for sentence in split_sentences(r.body):
            if quotes.usable(sentence) and quotes.score(sentence, seat) > best_score:  # rows run newest first: ties keep the newest
                best = {"text": sentence, "cite": cite({"date": r.date, "page": r.page}), "subject": TOPIC_NAMES[top]}
                best_score = quotes.score(sentence, seat)
    return best


@st.cache_data(show_spinner=False)
def tour_stops() -> list:
    """The tour: one MP for each state and territory, mixing the sides of politics as evenly as the record allows.

    Each is a sitting MP (the latest speaker for a seat that exists at the 2025 election, who has spoken since that election),
    the one with the most speeches on the side used least so far; each stop has one short sentence of theirs from the record.
    """
    if _shapes() is None:
        return []
    mps = load_mp_list()
    _, latest = load_lookup()
    current = latest[pd.to_datetime(latest["last_seen"]) >= ELECTION_2025][["electorate", "mp_key"]]
    pool = current.merge(mps[["mp_key", "name", "party", "n"]], on="mp_key")
    pool["state"] = pool["electorate"].map(lambda e: locate(e)[1] if locate(e)[0] else None)
    used, stops = Counter(), []
    for state in TOUR_ORDER:
        here = pool[pool["state"] == state].assign(side=lambda d: d["party"].map(bloc))
        for r in sorted(here.itertuples(), key=lambda r: (used[r.side], -r.n)):
            quote = _tour_quote(r.mp_key, r.electorate)
            if quote:
                last, _, first = re.sub(r"\s*\([^)]*\)", "", r.name).partition(", ")  # "Katter, Bob (Jr)" -> "Bob Katter"
                name = f"{first} {last}".strip()
                stops.append({"scene": _scene(r.electorate), "quote": quote, "who": name, "party": PARTY_NAMES.get(r.party, r.party)})
                used[r.side] += 1
                break
    return stops
