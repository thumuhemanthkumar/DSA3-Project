"""Turning raw lyric text into a clean token stream.

Two songs can say the same thing with different punctuation, casing, or
padding ("Oh-oh-ohhh!"), so every document goes through the same pipeline
before it is ever hashed:

    lowercase -> strip punctuation -> collapse elongations -> tokenise

Each token keeps the line it came from so that a match can be reported as
"lines 4-7 of the query" instead of "tokens 38-71".
"""

from collections import namedtuple

Token = namedtuple("Token", ["text", "line_no"])

_PUNCT = set(".,;:!?\"'`()[]{}<>*_~/\\|@#$%^&+=")

# Filler syllables that appear in almost every song and would otherwise
# create fake matches between unrelated tracks.
FILLERS = {
    "oh", "ooh", "ohh", "ah", "aah", "yeah", "yea", "hey", "hmm", "mmm",
    "la", "na", "da", "woah", "whoa", "uh", "huh",
}


def _squash_elongation(word):
    """"soooo" -> "soo"; keeps doubles, kills runs of 3+."""
    out = []
    run = 0
    prev = ""
    for ch in word:
        if ch == prev:
            run += 1
        else:
            run = 1
            prev = ch
        if run <= 2:
            out.append(ch)
    return "".join(out)


def normalize_line(line):
    cleaned = []
    for ch in line.lower():
        if ch in _PUNCT:
            cleaned.append(" ")
        elif ch == "-":
            cleaned.append(" ")
        else:
            cleaned.append(ch)
    return "".join(cleaned)


def tokenize(text, drop_fillers=True):
    """Return a list of Token(text, line_no). Line numbers are 1-based."""
    tokens = []
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        for word in normalize_line(raw_line).split():
            word = _squash_elongation(word)
            if not word:
                continue
            if drop_fillers and word in FILLERS:
                continue
            tokens.append(Token(word, line_no))
    return tokens


def token_texts(tokens):
    return [t.text for t in tokens]


def line_span(tokens, start, end):
    """Line range covered by tokens[start:end+1], clamped to the document."""
    if not tokens:
        return (0, 0)
    start = max(0, min(start, len(tokens) - 1))
    end = max(0, min(end, len(tokens) - 1))
    return (tokens[start].line_no, tokens[end].line_no)
