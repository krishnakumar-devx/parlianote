"""Drive the Streamlit app headlessly with unusual inputs and check it neither crashes nor misleads.

Run from the project root:  python tests/test_app_edge_cases.py
"""

import sys
import pandas as pd
import pyarrow.parquet as pq
from streamlit.testing.v1 import AppTest

APP = "app/app.py"
results = []


def check(name, cond, detail=""):
    """Print and record one PASS or FAIL result."""
    results.append((name, bool(cond), detail))
    print(
        ("PASS " if cond else "FAIL ")
        + name.encode("ascii", "replace").decode()
        + (f"   [{detail.encode('ascii', 'replace').decode()}]" if detail else ""),
        flush=True,
    )


def fresh():
    """A freshly started copy of the app."""
    return AppTest.from_file(APP, default_timeout=300).run()


def texts(at):
    """All visible text on the page (messages, markdown and captions) joined into one string."""
    out = (
        [e.value for e in at.warning]
        + [e.value for e in at.error]
        + [e.value for e in at.info]
        + [m.value for m in at.markdown]
        + [c.value for c in at.caption]
    )
    return " ".join(out)


def set_view(at, view):
    # radio[0] is the mode, radio[1] the view selector (only present once an MP is chosen)
    """Switch to one of the single-MP screens."""
    at.radio[1].set_value(view).run()
    return at


pc = pd.read_parquet("app/data/postcodes.parquet")
mps = pq.read_table("app/data/speeches.parquet", columns=["mp_key", "displayName"]).to_pandas().groupby("displayName").size().sort_values()
multi = pc.groupby("postcode")["electorate"].nunique()
multi_pc = multi[multi > 1].index[0]
few_mp = mps.index[0]

# ---- postcode box
at = fresh()
check("loads with no exception", not at.exception)
check("home screen explains the 3 steps and offers examples", "Enter your postcode" in texts(at) and "example postcode" in texts(at))
for bad in ["abc", "123", "20 8", "0000", "1234", "9999", "0200", "7999"]:
    at = fresh()
    at.text_input[0].set_value(bad).run()
    shown = texts(at)
    ok = (not at.exception) and any(k in shown for k in ["4-digit", "could not match", "PO boxes"])
    check(f"postcode {bad!r} handled", ok)
at = fresh()
at.text_input[0].set_value("2088").run()
check("valid postcode 2088 gives Steggall", (not at.exception) and "Steggall" in texts(at))
at = fresh()
at.text_input[0].set_value(multi_pc).run()
check(
    f"postcode spanning several areas ({multi_pc}) asks user to choose",
    (not at.exception) and any("more than one local area" in s.label for s in at.selectbox),
    f"{len(at.selectbox)} selectboxes",
)

# ---- every view for a normal MP and for the MP with the fewest speeches
for who, setter in [
    ("Steggall (postcode 2088)", lambda a: a.text_input[0].set_value("2088")),
    (f"fewest speeches: {few_mp}", lambda a: a.selectbox[0].set_value(few_mp)),
]:
    at = fresh()
    setter(at).run()
    for v in ["Overview", "Ask the record", "Question Time", "Promises", "About and limits"]:
        set_view(at, v)
        check(f"{who}: view {v!r} renders without error", not at.exception, str(at.exception[0].value)[:80] if at.exception else "")

# ---- Ask the record (one MP)
at = fresh()
at.text_input[0].set_value("2088").run()
set_view(at, "Ask the record")
ask = lambda q: (at.text_input[1].set_value(q).run(), texts(at))[1]
for q, want, label in [
    ("climate change", ["Key sentences"], "relevant topic gives key sentences"),
    ("banana bread recipe", ["couldn't find anything"], "off-topic gives a 'no clear match' message"),
    ("quantum computing", ["couldn't find anything"], "another off-topic query handled"),
    ("a", ["few words"], "single letter asks for more words"),
    ("???", ["few words"], "punctuation only asks for more words"),
    ("12345", ["few words"], "numbers only asks for more words"),
    ("<script>alert(1)</script>", ["couldn't find anything", "few words", "Key sentences"], "HTML in the query is harmless"),
    ("x" * 400, ["couldn't find anything", "few words", "Key sentences"], "400-character query is capped"),
    ("what has she said about gambling ads?", ["Key sentences"], "full question works"),
    ("更多的住房", ["few words"], "non-English text handled"),
]:
    shown = ask(q)
    check(f"Ask the record: {label}", (not at.exception) and any(w in shown for w in want), q[:25])
shown = ask("")
check("Ask the record: empty query shows nothing odd", not at.exception)

# ---- Search all MPs
at = fresh()
at.radio[0].set_value("Search all MPs").run()
sq = lambda q: (at.text_input[0].set_value(q).run(), texts(at))[1]
check("Search all MPs opens with steps and example topics", (not at.exception) and "Try one of these" in texts(at))
for q, want, label in [
    ("what has been said about housing affordability?", ["Who spoke about this"], "full question gives named speakers"),
    ("climate change", ["Who spoke about this"], "topic gives named speakers"),
    ("banana bread recipe", ["couldn't find anything"], "off-topic says nothing clearly matches"),
    ("how to train a puppy", ["couldn't find anything"], "another off-topic query"),
    ("asdfghjkl", ["couldn't find anything"], "gibberish says nothing clearly matches"),
    ("a", ["few words"], "single letter asks for more words"),
    ("!!!", ["few words"], "punctuation only asks for more words"),
    ("the of and", ["couldn't find anything", "few words", "Who spoke about this"], "only filler words handled"),
    ("<b>bold</b> climate", ["Who spoke about this", "couldn't find anything"], "HTML in the query is harmless"),
    ("x" * 400, ["couldn't find anything", "few words", "Who spoke about this"], "400-character query is capped"),
]:
    shown = sq(q)
    check(f"Search all: {label}", (not at.exception) and any(w in shown for w in want), q[:25])
shown = sq("climate change")
check("Search all: named speakers include a party and area", ("ALP" in shown or "LIB" in shown or "IND" in shown or "GRN" in shown))
at.slider[0].set_value((2020, 2020)).run()
check("Search all: single-year filter does not crash", not at.exception)

bad = [r for r in results if not r[1]]
print(f"\n{len(results) - len(bad)} of {len(results)} checks passed")
for r in bad:
    print("  FAILED:", r[0], r[2])
sys.exit(1 if bad else 0)
