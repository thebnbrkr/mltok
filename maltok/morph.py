"""Word-internal boundaries from mlmorph (Thottingal's FST analyser): stem | suffix, and compound joins.

Single stem. For a word whose best analysis is one root with POS n, np or v followed only by tags
(e.g. മരം<n><locative>, പ്രഖ്യാപിക്കുക<v><past>), the stem ends at the longest common prefix of the
word and its lemma. Two versions of that boundary are kept:
  exact    the prefix length; may fall inside an akshara (തരംതിരിക്ക|ാൻ)
  rounded  moved back to the start of its akshara (വീട്/വീട്ടിൽ -> വീ|ട്ടിൽ)
`cut` is exact, unless that would split a consonant cluster (next to a virama), then rounded.

Compounds and sandhi (സമാസം, സന്ധി; what പിരിച്ചെഴുതുക undoes). When the analysis has several roots
(ആന<n><adj>പുറം<n><locative>, കാണുക<v><past>ഇല്ല<neg>), each later root is located in the word by the start
of its surface form: the lemma without the verb ending ‑ുക and without a final anusvara, chillu or virama,
with an initial vowel written as its vowel sign (ഇല്ല -> ില്ല) or dropped (അ). The join is where that form
starts, so a doubled consonant (ആന|പ്പുറം) or an inserted glide യ/വ (മഴ|യില്ല) stays with the following part. A join that
would split a consonant cluster moves back to the start of its akshara. If the last root is an inflected
noun or verb, its stem | suffix boundary is added too (ആന | പ്പുറ | ത്തു). All of these are silver.
"""
from __future__ import annotations

import os
import re
from multiprocessing import Pool

from .text import MALAYALAM_RUN, VIRAMA, ZWJ, ZWNJ, akshara_starts

MARK = "⁣"   # INVISIBLE SEPARATOR: marks a forced boundary; the tokenizer splits on it and drops it
_SINGLE = re.compile(r"([^<]+)<(n|np|v)>((?:<[^>]+>)+)")
_ROOTS = re.compile(r"([^<]+)((?:<[^>]+>)+)")
_VOWEL_SIGN = {"ആ": "ാ", "ഇ": "ി", "ഈ": "ീ", "ഉ": "ു", "ഊ": "ൂ", "ഋ": "ൃ", "എ": "െ", "ഏ": "േ",
               "ഐ": "ൈ", "ഒ": "ൊ", "ഓ": "ോ", "ഔ": "ൌ"}
_FINAL = "ംൺൻർൽൾൿ" + VIRAMA     # anusvara, chillus, virama
_analyser = None
_cache: dict = {}            # word -> best analysis string or None (the slow mlmorph call)


def _get():
    global _analyser
    if _analyser is None:
        from mlmorph import Analyser
        _analyser = Analyser()
    return _analyser


def _analyse(word: str) -> str | None:
    res = _get().analyse(word)
    return res[0][0] if res else None


def analysis(word: str) -> str | None:
    if word not in _cache:
        _cache[word] = _analyse(word)
    return _cache[word]


def _chunk(words):
    return [(w, _analyse(w)) for w in words]


def precompute(words, processes: int | None = None, chunk: int = 2000) -> int:
    """Fill the analysis cache for many words in parallel (mlmorph is the slow step)."""
    todo = [w for w in set(words) if w and w not in _cache]
    parts = [todo[i:i + chunk] for i in range(0, len(todo), chunk)]
    with Pool(processes or os.cpu_count()) as pool:
        for part in pool.imap_unordered(_chunk, parts):
            _cache.update(part)
    return len(todo)


def _lcp(a: str, b: str) -> int:
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def _safe(word: str, b: int, starts: set[int]) -> int:
    """b, or the start of its akshara if b would split a consonant cluster."""
    if 0 < b < len(word) and (word[b - 1] == VIRAMA or word[b] in (VIRAMA, ZWJ, ZWNJ)):
        return max(s for s in starts if s <= b)
    return b


def boundary(word: str) -> tuple | None:
    """Single-stem words only: (pos, exact, rounded, cut, tags), or None."""
    a = analysis(word)
    m = _SINGLE.fullmatch(a) if a else None
    if not m:
        return None
    lemma, pos, tags = m.groups()
    lcp = _lcp(word, lemma)
    starts = akshara_starts(word)
    rounded = max(s for s in starts if s <= lcp)
    cut = _safe(word, lcp, starts)
    if not (0 < cut < len(word) and 0 < rounded < len(word)):
        return None
    return ("verb" if pos == "v" else "noun", lcp, rounded, cut, tags)


def surface_stem(lemma: str, verb: bool = False) -> str:
    """How a root usually starts when it is not first in a word (see module docstring)."""
    s = lemma[:-2] if verb and lemma.endswith("ുക") else lemma
    s = s.rstrip(_FINAL)
    if s[:1] == "അ":
        s = s[1:]
    elif s[:1] in _VOWEL_SIGN:
        s = _VOWEL_SIGN[s[0]] + s[1:]
    return s


def find_join(word: str, later_part: str, after: int, verb: bool = False) -> int | None:
    """Where `later_part` starts in `word`, searching after position `after` (exact, may be in an akshara)."""
    stem = surface_stem(later_part, verb)
    if not stem:
        return None
    i = word.find(stem[:2], after + 1)
    return i if 0 < i < len(word) else None


def compound(word: str) -> tuple | None:
    """Multi-root words: ([joins as (exact, rounded)], cuts used for segmentation), or None."""
    a = analysis(word)
    if not a:
        return None
    roots = _ROOTS.findall(a)
    if len(roots) < 2 or "".join(r + t for r, t in roots) != a:
        return None
    starts = akshara_starts(word)
    joins, cuts, prev = [], [], 0
    for (before, _), (lemma, tags) in zip(roots, roots[1:]):
        b = find_join(word, lemma, prev, verb=tags.startswith("<v>"))
        if b is None:
            return None
        if word[b - 1] in "യവ" and not before.rstrip(_FINAL).endswith(word[b - 1]):
            b -= 1                       # inserted glide (ആഗമം) belongs to the following part: മഴ | യില്ല
        joins.append((b, max(s for s in starts if s <= b)))
        cuts.append(_safe(word, b, starts))
        prev = b
    last, ltags = roots[-1]
    pos = re.match(r"<(n|np|v)>", ltags)
    if pos and len(re.findall(r"<[^>]+>", ltags)) > 1:     # inflected last root: add its stem | suffix
        stem = surface_stem(last, verb=pos.group(1) == "v")
        s = joins[-1][0]
        end = s + _lcp(word[s:], stem)
        if s < end < len(word):
            cuts.append(_safe(word, end, starts))
    cuts = sorted({c for c in cuts if 0 < c < len(word)})
    return (joins, cuts) if cuts else None


def segment_word(word: str, compounds: bool = False) -> str:
    cuts = []
    if compounds:
        c = compound(word)
        if c:
            cuts = c[1]
    if not cuts:
        b = boundary(word)
        cuts = [b[3]] if b else []
    for c in sorted(cuts, reverse=True):
        word = word[:c] + MARK + word[c:]
    return word


def segment_text(text: str, compounds: bool = False) -> str:
    return MALAYALAM_RUN.sub(lambda m: segment_word(m.group(), compounds), text)
