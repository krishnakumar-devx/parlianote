"""Political blocs and Speaker names, so 'friendly' questions are detected correctly.

A Dorothy Dixer is a question from a government-bloc MP to a government minister. The Coalition is two parties
(Liberal and National), so exact party matching wrongly treats a Liberal asking a National minister as hostile.
"""

BLOC = {"ALP": "Labor", "LIB": "Coalition", "NP": "Coalition", "NAT": "Coalition", "LNP": "Coalition", "CLP": "Coalition"}


def bloc(party) -> str:
    """Map a party code (ALP, LIB, NP, ...) to its political bloc: Labor, Coalition or Other."""
    return BLOC.get(str(party), "Other")


# Speakers of the House 2011-2025: their 'answers' are rulings, not ministerial answers.
CHAIR = {
    "Jenkins, Harry (Jr)",
    "Slipper, Peter",
    "Burke, Anna",
    "Bishop, Bronwyn",
    "Smith, Tony (Anthony David)",
    "Wallace, Andrew",
    "Dick, Milton",
}
