"""Settings and wording shared across the web app, kept in one place so they are easy to find and change."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "app" / "data"  # pre-built files, created by src/prepare_app_data.py

HANSARD_HOME = "https://www.aph.gov.au/Parliamentary_Business/Hansard"
ELECTION_2025 = pd.Timestamp("2025-05-03")  # seats may have changed hands after this date
MAX_QUERY_CHARS = 300  # longest search text we accept

# The two top-level modes, and the screens available once an MP has been chosen.
MODES = ["Find my MP", "Search all MPs", "Compare MPs", "Parliament trends"]
VIEWS = ["Overview", "Ask the record", "Question Time", "Promises", "About and limits"]

# Full party names for the abbreviations used in the record.
PARTY_NAMES = {
    "ALP": "Labor",
    "LIB": "Liberal",
    "NP": "Nationals",
    "NAT": "Nationals",
    "NATS WA": "Nationals WA",
    "IND": "Independent",
    "GRN": "Greens",
    "KAP": "Katter's Australian Party",
    "CA": "Centre Alliance",
    "NXT": "Nick Xenophon Team",
    "CLP": "Country Liberal Party",
    "PUP": "Palmer United Party",
    "CDP": "Christian Democratic Party",
    "PHON": "One Nation",
}

# One-click examples shown on the home screen.
EXAMPLE_POSTCODES = [("2088", "Mosman"), ("2000", "Sydney"), ("3000", "Melbourne"), ("4000", "Brisbane"), ("6000", "Perth")]
EXAMPLE_TOPICS = ["climate change", "housing affordability", "gambling advertising", "aged care", "paid parental leave", "drought"]

NO_MATCH_TIPS = (
    "Try different or simpler words, widen the years, or check the spelling. "
    "The record only covers speeches in the House of Representatives up to July 2025, "
    "so a topic discussed elsewhere (the Senate, interviews, social media) will not appear here."
)
