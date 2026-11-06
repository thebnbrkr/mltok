"""Stem | suffix boundaries from mlmorph (Thottingal's FST analyser).

For a word whose best analysis is a single root with POS n, np or v followed only by tags
(e.g. മരം<n><locative>, പ്രഖ്യാപിക്കുക<v><past>), the stem ends at the longest common prefix of
the word and its lemma. Two versions of that boundary are kept:
  exact    the prefix length; may fall inside an akshara (തരംതിരിക്ക|ാൻ)
  rounded  moved back to the start of its akshara (വീട്/വീട്ടിൽ -> വീ|ട്ടിൽ)
`cut` is the boundary the morph-guided tokenizer is forced to respect: exact, unless that would
split a consonant cluster (next to a virama), then rounded. These boundaries are silver, not gold.
"""
from __future__ import annotations

import os
import re
from multiprocessing import Pool

from .text import MALAYALAM_RUN, VIRAMA, ZWJ, ZWNJ, akshara_starts

MARK = "⁣"   # INVISIBLE SEPARATOR: marks a forced boundary; the tokenizer splits on it and drops it
_ANALYSIS = re.compile(r"([^<]+)<(n|np|v)>((?:<[^>]+>)+)")
_analyser = None
_cache: dict = {}


def _get():
    global _analyser
    if _analyser is None:
        from mlmorph import Analyser
        _analyser = Analyser()
    return _analyser


def _compute(word: str) -> tuple | None:
    res = _get().analyse(word)
    if not res:
        return None
    m = _ANALYSIS.fullmatch(res[0][0])
    if not m:
        return None
    lemma, pos, tags = m.groups()
    lcp = 0
    while lcp < min(len(word), len(lemma)) and word[lcp] == lemma[lcp]:
        lcp += 1
    rounded = max(s for s in akshara_starts(word) if s <= lcp)
    splits_cluster = 0 < lcp < len(word) and (word[lcp - 1] == VIRAMA or word[lcp] in (VIRAMA, ZWJ, ZWNJ))
    cut = rounded if splits_cluster else lcp
    if not (0 < cut < len(word) and 0 < rounded < len(word)):
        return None
    return ("verb" if pos == "v" else "noun", lcp, rounded, cut, tags)


def boundary(word: str) -> tuple | None:
    """(pos, exact, rounded, cut, tags), or None without a usable single-stem analysis."""
    if word not in _cache:
        _cache[word] = _compute(word)
    return _cache[word]


def _chunk(words):
    return [(w, _compute(w)) for w in words]


def precompute(words, processes: int | None = None, chunk: int = 2000) -> int:
    """Fill the cache for many words in parallel (mlmorph is the slow step)."""
    todo = [w for w in set(words) if w and w not in _cache]
    parts = [todo[i:i + chunk] for i in range(0, len(todo), chunk)]
    with Pool(processes or os.cpu_count()) as pool:
        for part in pool.imap_unordered(_chunk, parts):
            _cache.update(part)
    return len(todo)


def segment_word(word: str) -> str:
    b = boundary(word)
    return word if b is None else word[:b[3]] + MARK + word[b[3]:]


def segment_text(text: str) -> str:
    return MALAYALAM_RUN.sub(lambda m: segment_word(m.group()), text)
