"""Shared LDA helpers (same preprocessing as notebooks/parlianote_analysis.ipynb, Section 4)."""

import re
import nltk
from nltk.corpus import stopwords

try:
    _STOP = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    _STOP = set(stopwords.words("english"))

# Words that are frequent in Parliament but say nothing about a topic.
PARL_STOP = {
    "member",
    "members",
    "house",
    "government",
    "minister",
    "speaker",
    "mr",
    "mrs",
    "ms",
    "dr",
    "honourable",
    "hon",
    "today",
    "would",
    "also",
    "said",
    "one",
    "us",
    "get",
    "going",
    "want",
    "know",
    "think",
    "like",
    "time",
    "people",
    "australia",
    "australian",
    "australians",
    "parliament",
    "bill",
    "motion",
    "committee",
    "deputy",
    "president",
    "senate",
    "point",
    "question",
    "answer",
    "matter",
    "issue",
    "way",
    "make",
    "made",
    "take",
    "taken",
    "many",
    "much",
    "well",
    "actually",
}
STOP = _STOP | PARL_STOP
TOKEN_RE = re.compile(r"[a-zA-Z]{3,}")


def tokenize(text: str):
    """Lower-case alphabetic words with stop words and parliamentary filler words removed."""
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOP]


# human-readable names chosen from each topic's top words (see notebook Section 4)
TOPIC_NAMES = {
    0: "Regional & local infrastructure",
    1: "Health, aged care & energy",
    2: "Community & family life",
    3: "Legislation & amendments",
    4: "Defence & veterans",
    5: "Economy, tax & budget",
    6: "Partisan debate",
    7: "Industry, trade & agriculture",
    8: "Rights, safety & security",
    9: "Education & training",
}
