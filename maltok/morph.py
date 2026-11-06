"""Word-internal boundaries from mlmorph (Thottingal's FST analyser): stem | suffix, compound joins, and a
dictionary-checked sandhi fallback for words mlmorph does not know.

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
starts, so a doubled consonant (ആന|പ്പുറം) or an inserted glide യ/വ (മഴ|യില്ല) stays with the following part.
A join that would split a consonant cluster moves back to the start of its akshara. If the last root is an
inflected noun or verb, its stem | suffix boundary is added too (ആന | പ്പുറ | ത്തു).

mlmorph ranks analyses with fewer parts first, so a word that is a word by itself stays whole (കുട്ടി; also
lexicalised compounds such as തീപ്പെട്ടി, which it also knows as തീ + പെട്ടി).

Fallback (words mlmorph cannot analyse at all, e.g. കാട്ടാന): the sandhi formulas propose splits, each with the
parts restored (ട്ട|ാ -> ട് + ആ, doubled consonant -> single, glide dropped, ന്മ -> ൽ + മ, ...); a split is
accepted only if both parts are known words (mlmorph analyses them, or they occur at least LEX_MIN times as
words in the training corpus). Among accepted splits the most balanced one wins; otherwise the word stays
whole. Two parts at most. All of these boundaries are silver.
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
_SIGN_VOWEL = {v: k for k, v in _VOWEL_SIGN.items()}
_CHILLU = {"ണ": "ൺ", "ന": "ൻ", "ര": "ർ", "ല": "ൽ", "ള": "ൾ", "ക": "ൿ"}
_SUBST = {"ന്മ": ("ൽ", "മ"), "ണ്ണ": ("ൺ", "ന"), "ന്ത": ("ം", "ത"), "മ്പ": ("ം", "പ")}   # ആദേശം, reversed
_FINAL = "ംൺൻർൽൾൿ" + VIRAMA     # anusvara, chillus, virama
_CONS = {chr(c) for c in range(0x0D15, 0x0D3B)}
LEX_MIN = 5
_analyser = None
_cache: dict = {}            # word -> best analysis string or None (the slow mlmorph call)
_fb_cache: dict = {}         # word -> (cut, rule, left, right) or None, for words mlmorph cannot analyse
_lexicon: set = set()        # corpus words seen at least LEX_MIN times (set_lexicon)


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


def set_lexicon(words) -> None:
    """Known corpus words for the fallback (an iterable of words already filtered by frequency)."""
    global _lexicon
    _lexicon = set(words)


def known(part: str) -> bool:
    if len(part) < 2:
        return False
    if part in _lexicon:
        return True
    a = analysis(part)
    return bool(a) and not a.endswith("<fw>")


def candidates(word: str) -> list[tuple[int, str, str, str]]:
    """(surface cut, rule, restored left part, restored right part) for every place a sandhi formula fits."""
    out, n = [], len(word)
    for i in range(1, n - 1):            # restored parts must still be 2+ characters (checked later)
        c, prev = word[i], word[i - 1]
        if c in _SIGN_VOWEL and prev in _CONS:                      # C·Vsign: the vowel began the next word
            right = _SIGN_VOWEL[c] + word[i + 1:]
            out += [(i, "lopa", word[:i] + VIRAMA, right), (i, "lopa-u", word[:i] + "ു", right)]
            if prev in _CHILLU:
                out.append((i, "chillu", word[:i - 1] + _CHILLU[prev], right))
            if prev == "മ":
                out.append((i, "anusvara", word[:i - 1] + "ം", right))
        if c in "യവ" and word[i + 1:i + 2] in _SIGN_VOWEL and prev != VIRAMA:   # ആഗമം: inserted glide
            out.append((i, "agama", word[:i], _SIGN_VOWEL[word[i + 1]] + word[i + 2:]))
        if c in _CONS and word[i + 1:i + 2] == VIRAMA and word[i + 2:i + 3] == c and prev != VIRAMA:  # ദ്വിത്വം
            out.append((i, "dvitva", word[:i], word[i + 2:]))
            if word[i + 3:i + 4] in _SIGN_VOWEL:                     # doubling + elision: കാ|ട്ടാന -> കാട് + ആന
                out.append((i, "dvitva-lopa", word[:i] + c + VIRAMA, _SIGN_VOWEL[word[i + 3]] + word[i + 4:]))
        for cluster, (l, r) in _SUBST.items():
            if word.startswith(cluster, i):
                out.append((i, "adesa", word[:i] + l, r + word[i + len(cluster):]))
        if (c in _CONS or c in _VOWEL_SIGN) and (prev in _FINAL[:-1] or prev in _SIGN_VOWEL) and word[i + 1:i + 2] != VIRAMA:
            out.append((i, "plain", word[:i], word[i:]))             # word-final letter, then a new word
    return out


def _fallback(word: str) -> tuple | None:
    best = None
    for cut, rule, left, right in candidates(word):
        if len(left) >= 2 and len(right) >= 2 and known(left) and known(right):
            score = min(len(left), len(right))
            if best is None or score > best[0]:
                best = (score, cut, rule, left, right)
    return best[1:] if best else None


def fallback(word: str) -> tuple | None:
    """(cut, rule, left, right) for a word mlmorph cannot analyse, or None (keep whole)."""
    if word not in _fb_cache:
        _fb_cache[word] = _fallback(word) if analysis(word) is None else None
    return _fb_cache[word]


def _chunk(args):
    words, with_fallback = args
    out = []
    for w in words:
        a = _analyse(w)
        _cache[w] = a
        out.append((w, a, _fallback(w) if with_fallback and a is None else None))
    return out


def precompute(words, processes: int | None = None, chunk: int = 2000, with_fallback: bool = False) -> int:
    """Fill the caches for many words in parallel (mlmorph is the slow step). The fallback uses the lexicon
    set before this call (worker processes inherit it)."""
    todo = [w for w in set(words) if w and (w not in _cache or (with_fallback and w not in _fb_cache))]
    parts = [(todo[i:i + chunk], with_fallback) for i in range(0, len(todo), chunk)]
    with Pool(processes or os.cpu_count()) as pool:
        for part in pool.imap_unordered(_chunk, parts):
            for w, a, fb in part:
                _cache[w] = a
                if with_fallback:
                    _fb_cache[w] = fb
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


def compound(word: str, use_fallback: bool = False) -> tuple | None:
    """([joins as (exact, rounded)], cuts used for segmentation), or None. Multi-root mlmorph analyses; with
    use_fallback, also the dictionary-checked sandhi split of a word mlmorph cannot analyse."""
    a = analysis(word)
    starts = akshara_starts(word)
    if not a:
        fb = fallback(word) if use_fallback else None
        if not fb:
            return None
        cut = fb[0]
        return [(cut, max(s for s in starts if s <= cut))], [_safe(word, cut, starts)]
    roots = _ROOTS.findall(a)
    if len(roots) < 2 or "".join(r + t for r, t in roots) != a:
        return None
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


def segment_word(word: str, compounds: bool = False, use_fallback: bool = False) -> str:
    cuts = []
    if compounds:
        c = compound(word, use_fallback)
        if c:
            cuts = c[1]
    if not cuts:
        b = boundary(word)
        cuts = [b[3]] if b else []
    for c in sorted(cuts, reverse=True):
        word = word[:c] + MARK + word[c:]
    return word


def segment_text(text: str, compounds: bool = False, use_fallback: bool = False) -> str:
    return MALAYALAM_RUN.sub(lambda m: segment_word(m.group(), compounds, use_fallback), text)
