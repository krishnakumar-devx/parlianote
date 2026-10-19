"""Picking a sentence from the record to show on its own (the idle tour, "On this day"): short, self-contained and non-partisan.

On a public screen the quote should be about places and work, not the contest between parties, and must make sense without the
rest of the speech. These rules are deliberately strict: it is better to show nothing than a misleading fragment.
"""

import re

from src.evasion import ATTACK

# Procedure, thanks and interruptions.
PROCEDURAL = re.compile(
    r"\b(speaker|honourable|member for|motion|i move|i second|i rise|i thank|i commend|order|interject|second reading|"
    r"this bill|the bill|amendment|chair)\b",
    re.I,
)
# Parties, sides and office-holders, and unnamed people ("she", "he") who need the rest of the speech to make sense.
POLITICAL = re.compile(
    r"\b(labor|liberals?|coalition|lnp|greens?|nationals?|one nation|government|opposition|those opposite|prime minister|"
    r"ministers?|treasurer|shadow|part(y|ies)|elections?|vote|voters?|she|he|her|him|his)\b",
    re.I,
)
# Deaths, tragedies, abuse and violence are part of the record, but not for a cheerful card on a public screen.
SOMBRE = re.compile(
    r"\b(died|dies|death|deaths|dead|killed|murder\w*|funeral|condolence\w*|traged\w*|tragic|minute's silence|passed away|grie\w*|"
    r"mourn\w*|victims?|abuse\w*|sexual|violen\w*|assault\w*|rape|suicid\w*|self-harm|terror\w*|war crimes?)\b",
    re.I,
)
LOCAL = re.compile(r"\b(electorate|community|communities|local|region|regional|town|families|residents|constituents)\b", re.I)


def usable(sentence: str) -> bool:
    """Whether a sentence can stand alone on a public screen."""
    return (
        14 <= len(sentence.split()) <= 30
        and sentence.endswith(".")
        and sentence[:1].isupper()
        and not re.match(r"(Well|So|And|But|I, too|Yes|No)\b", sentence)
        and not any(c in sentence for c in "?\"“”")
        and not PROCEDURAL.search(sentence)
        and not POLITICAL.search(sentence)
        and not ATTACK.search(sentence)
        and not SOMBRE.search(sentence)
    )


def score(sentence: str, place: str = "") -> int:
    """Higher for sentences about a place or community (or naming `place`) and for concrete numbers."""
    local = bool(LOCAL.search(sentence) or (place and place.lower() in sentence.lower()))
    return 2 * local + bool(re.search(r"\d", sentence))
