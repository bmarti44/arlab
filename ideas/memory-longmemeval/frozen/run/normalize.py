"""Frozen answer normalization + alias extraction (shared by PREPARE and the evaluator; the agent never sees it)."""
import re
import string

NUM = {w: str(i) for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen "
                                          "fourteen fifteen sixteen seventeen eighteen nineteen twenty".split())}
NUM.update({"thirty": "30", "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70", "eighty": "80", "ninety": "90",
            "hundred": "100", "first": "1st", "second": "2nd", "third": "3rd", "once": "1 time", "twice": "2 times"})
ARTICLES = {"a", "an", "the"}
TENS = {"20", "30", "40", "50", "60", "70", "80", "90"}
ABSTAIN = ["i dont know", "unknown", "not mentioned", "you did not mention this", "you did not mention this information",
           "no information", "not enough information", "cannot be determined", "i do not know"]


def normalize(s) -> str:
    s = str(s).lower().replace("’", "'").replace("'", "")
    s = re.sub(r"(?<=\d),(?=\d{3})", "", s)                       # 1,000 → 1000
    s = "".join(" " if c in string.punctuation and c not in ":$%" else c for c in s)
    s = s.replace("$", " $ ").replace("%", " % ")
    toks = []
    for t in (NUM.get(t, t) for t in s.split() if t not in ARTICLES):
        if toks and t.isdigit() and len(t) == 1 and toks[-1] in TENS:   # twenty one → 21
            toks[-1] = str(int(toks[-1]) + int(t))
        else:
            toks.append(t)
    return " ".join(toks)


ABSTAIN_NORM = {normalize(p) for p in ABSTAIN}


def aliases(answer) -> list[str]:
    """Gold variants from the dataset's own answer text: 'X. Y (including the last day) is also acceptable.' and 'X (or Y)'."""
    a = str(answer).strip()
    out = []
    m = re.match(r"^(.*?)\.\s+(.*?)\s*\(including the last day\) is also acceptable\.?$", a)
    if m:
        out += [m.group(1), m.group(2)]
    else:
        m = re.match(r"^(.*?)\s*\(or ([^)]*)\)\.?$", a)
        out += [m.group(1), m.group(2)] if m else [a]
    return sorted({normalize(x) for x in out if normalize(x)})


def score(answer: str, gold: dict) -> float:
    """1 iff the normalized answer exactly equals a normalized gold alias (or an abstention phrase for abstention items)."""
    n = normalize(answer)
    if gold["abstention"]:
        return float(n in ABSTAIN_NORM)
    return float(n in set(gold["aliases"]))
