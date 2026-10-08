"""Rule-based detection of pledges in ministerial answers (classical approach)."""

import re

FIRM = re.compile(
    r"\b(we|i|the government|this government|our government|the coalition|labor|the minister)\s+"
    r"(will|shall|are going to|is going to|am going to|have committed to|has committed to|are committed to|"
    r"is committed to|am committed to|commit(s)? to|guarantee|promise|pledge|undertake)\b",
    re.I,
)
GUARANTEE = re.compile(
    r"\b(i|we) (can )?(guarantee|assure|promise|commit)\b|\bi give (the house|an?) (assurance|undertaking)\b|"
    r"\b(we|i) (will not|won't)\b",
    re.I,
)
HEDGE = re.compile(
    r"\b(consider(ing)?|look(ing)? (at|into)|examin(e|ing)|review(ing)?|in due course|"
    r"will be (receiving|taking) advice|take (that|it) on notice|we'll see|wait for|if elected)\b",
    re.I,
)
# someone else's words ("he said: we will ...") are not a pledge by the speaker
REPORTED = re.compile(r"\b(he|she|they|you|\w+) (said|says|told|claimed|announced)\b|^on \d+ \w+,", re.I)
OWN_WORDS = re.compile(r"\b(i|we) (said|have said|told|promised)\b", re.I)
SPECIFIC = re.compile(
    r"(\d|\$|\bby (the end of )?(next|this|20\d\d)|\bfrom (1 )?(january|july|october)|"
    r"\bevery (week|year|month)|\bper cent\b)",
    re.I,
)


def split_sentences(text: str):
    """Split text into sentences, keeping only those of at least six words."""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'])", str(text).strip())
    return [p.strip() for p in parts if len(p.split()) >= 6]


def is_commitment(sentence: str) -> str:
    """Return 'C' (firm commitment), 'H' (hedged intention) or 'N' (not a commitment)."""
    if REPORTED.search(sentence) and not OWN_WORDS.search(sentence):
        return "N"
    if FIRM.search(sentence) or GUARANTEE.search(sentence):
        return "H" if HEDGE.search(sentence) else "C"
    if HEDGE.search(sentence) and re.search(r"\bwill\b", sentence, re.I):
        return "H"
    return "N"


def commitment_type(sentence: str) -> str:
    """'specific pledge' (flagged + concrete number/date/amount), 'general pledge' (flagged only) or 'none'."""
    if is_commitment(sentence) == "N":
        return "none"
    return "specific pledge" if SPECIFIC.search(sentence) else "general pledge"
