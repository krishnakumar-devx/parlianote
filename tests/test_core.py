"""Fast offline tests of the core logic behind the app (no models, no data files): python tests/test_core.py"""

import sys

import pandas as pd

sys.path.insert(0, ".")
from src import global_search as G
from src import retrieval as R
from src.blocs import CHAIR, bloc
from src.commitment import commitment_type, is_commitment, split_sentences
from src.stats import wilson
from src.ui.components import results_csv
from src.ui.data import cite
from src.ui.search import clean_query, tidy_query, usable_query

failures = []


def check(name, condition):
    """Print and record one PASS or FAIL result."""
    print(("PASS " if condition else "FAIL ") + name)
    if not condition:
        failures.append(name)


# --- political blocs: Liberal and National are one side, so a Liberal asking a National minister is a friendly question
check("Liberal and National are the same bloc", bloc("LIB") == bloc("NP") == bloc("NAT") == "Coalition")
check("Labor is its own bloc", bloc("ALP") == "Labor" and bloc("ALP") != bloc("LIB"))
check("independents and Greens are 'Other'", bloc("IND") == bloc("GRN") == "Other")
check("unknown or missing party does not crash", bloc(None) == "Other" and bloc("???") == "Other")
check("Speaker list includes the current Speaker", "Dick, Milton" in CHAIR)

# --- pledge detection
check("a specific pledge is recognised", commitment_type("We will deliver 1.2 million homes by 2029.") == "specific pledge")
check("a vague pledge is recognised", commitment_type("We will continue to work with the states.") == "general pledge")
check("a past fact is not a pledge", commitment_type("The superannuation guarantee went up to 12 per cent.") == "none")
check("someone else's quoted words are not a pledge", is_commitment("On 21 April, he said: 'We will bury the idea of a state.'") == "N")
check("a hedged intention is marked hedged", is_commitment("We will consider that in due course.") == "H")
check("'we will not' counts as a promise", is_commitment("We will not cut Medicare under this government.") != "N")
check(
    "very short sentences are dropped",
    split_sentences("Yes. We will build the new road by 2030. No.") == ["We will build the new road by 2030."],
)

# --- statistics
low, high = wilson(31, 40)
check("Wilson interval for 31/40 is about 63% to 88%", 0.62 < low < 0.64 and 0.87 < high < 0.89)
check("Wilson interval with no data is (0, 0)", wilson(0, 0) == (0.0, 0.0))
check("Wilson interval stays inside 0..1", wilson(0, 5)[0] == 0.0 and wilson(5, 5)[1] == 1.0)

# --- what the user typed
check("question filler is removed", clean_query("what has been said about housing affordability?") == "housing affordability")
check("a plain topic is unchanged", clean_query("climate change") == "climate change")
check("a query made only of filler keeps its original text", clean_query("what is the") == "what is the")
check("whitespace is tidied", tidy_query("  climate    change \n") == "climate change")
check("very long text is capped", len(tidy_query("x" * 1000)) == 300)
check("None is handled", tidy_query(None) == "")
check("words are usable", usable_query("aged care") and usable_query("a1b2cd"))
check("digits and punctuation alone are not usable", not usable_query("12345") and not usable_query("?!?") and not usable_query(""))

# --- 'is there a real answer?' cut-offs
check("MP search: strong, weak and none", [R.match_quality(x) for x in (6.0, 1.5, -3.0)] == ["strong", "weak", "none"])
check("MP search: missing or NaN score means none", R.match_quality(None) == "none" and R.match_quality(float("nan")) == "none")
check(
    "all-MPs search: strong, weak and none",
    [G.match_quality(a, b) for a, b in ((0.5, 5.0), (0.36, 2.0), (0.2, 6.0))] == ["strong", "weak", "none"],
)
check("all-MPs search: NaN score means none", G.match_quality(0.5, float("nan")) == "none")

# --- passages and exports
speeches = pd.DataFrame(
    {
        "speech_id": [1, 2],
        "date": pd.to_datetime(["2020-05-01", "2021-06-02"]),
        "page": [10.0, None],
        "speech_no": [3.0, 4.0],
        "words": [300, 10],
        "body": [" ".join(f"word{i}" for i in range(300)), "too short to use"],
    }
)
passages = R.make_passages(speeches)
check("short speeches are skipped when making passages", set(passages["speech_id"]) == {1})
check("long speeches become overlapping passages", len(passages) == 4 and all(len(t.split()) <= 110 for t in passages["text"]))
row = pd.Series({"date": pd.Timestamp("2008-02-13"), "page": 230.0})
check("citation includes the date and page", cite(row) == "House of Representatives Hansard, 13 Feb 2008, p. 230")
check(
    "citation copes with a missing page",
    cite(pd.Series({"date": pd.Timestamp("2008-02-13"), "page": float("nan")})).endswith("13 Feb 2008"),
)
csv = results_csv(
    pd.DataFrame({"displayName": ["Smith, A"], "party": ["ALP"], "date": ["2020-01-01"], "page": [5.0], "text": ["Hello, world"]}),
    speaker_col="displayName",
).decode()
check("CSV export has a header and quotes commas", csv.startswith("speaker,party,date,citation,text") and '"Hello, world"' in csv)

print(f"\n{'ALL' if not failures else len(failures)} {'passed' if not failures else 'FAILED'}")
sys.exit(1 if failures else 0)
