"""Frozen tools handed to the surface: a deterministic BM25 index. (No file, network or data access here.)"""
import math
import re
from collections import Counter

_TOK = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOK.findall(text.lower())


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.tfs = [Counter(tokenize(d)) for d in docs]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avg = sum(self.lens) / max(len(docs), 1)
        df = Counter(t for tf in self.tfs for t in tf)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.k1, self.b = k1, b

    def search(self, query: str, k: int = 10) -> list[int]:
        q = set(tokenize(query))
        scores = []
        for i, tf in enumerate(self.tfs):
            s = 0.0
            for t in q:
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.lens[i] / (self.avg or 1)))
            scores.append((s, -i))
        return [-i for s, i in sorted(scores, reverse=True)[:k] if s > 0]


class Tools:
    BM25 = BM25
    tokenize = staticmethod(tokenize)
