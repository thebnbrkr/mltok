"""The പിരിച്ചെഴുതുക gold set: does a tokenizer cut where a school split separates the parts?

gold/pirichezhuthuka.tsv lists joined words with their parts and sandhi kind. Each join is placed in the word
as maltok.morph places compound joins (where the next part's surface form starts; exact, or moved back to its
akshara); when the next part cannot be found (ആദേശം can change its first letter: കൺ + നീർ -> കണ്ണീർ), the
join falls after the longest common prefix with the previous part, at an akshara start. A word counts as
"join found" if the tokenizer cuts at the exact or the akshara-start position of every join, or, for rows with
whole_ok = yes (the joined form is itself a dictionary word), if it keeps the word as one piece; "clean" if in
addition it does not cut inside the first part.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from .morph import _lcp, find_join
from .text import akshara_starts, normalize

GOLD = Path(__file__).resolve().parent.parent / "gold" / "pirichezhuthuka.tsv"


def load(path: Path = GOLD) -> list[dict]:
    lines = [l for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]
    rows = list(csv.DictReader(lines, delimiter="\t"))
    for r in rows:
        r["word"] = normalize(r["word"])
        r["parts"] = [normalize(x.strip()) for x in r["parts"].split("+")]
    return rows


def joins(word: str, parts: list[str]) -> list[tuple[int, int]] | None:
    """(exact, akshara-start) position of each join."""
    starts = akshara_starts(word)
    out, prev = [], 0
    for before, part in zip(parts, parts[1:]):
        b = find_join(word, part, prev)
        if b is not None and word[b - 1] in "യവ" and not before.endswith(word[b - 1]):
            b -= 1
        if b is None:
            b = max(s for s in starts if s <= max(1, prev + _lcp(word[prev:], before)))
        if not 0 < b < len(word):
            return None
        out.append((b, max(s for s in starts if s <= b)))
        prev = b
    return out


def evaluate(pipe, rows: list[dict]) -> dict:
    from .evaluate import word_cuts
    per = defaultdict(lambda: [0, 0, 0])
    for r in rows:
        js = joins(r["word"], r["parts"])
        cuts, ok = word_cuts(pipe, r["word"])
        if js is None or not ok:
            continue
        whole = r.get("whole_ok") == "yes" and not cuts
        found = whole or all(e in cuts or a in cuts for e, a in js)
        clean = whole or (found and not any(0 < c < min(js[0]) for c in cuts))
        for key in (r["sandhi"], "all"):
            per[key][0] += 1
            per[key][1] += found
            per[key][2] += clean
    return {k: {"n": n, "found": f / n, "clean": c / n} for k, (n, f, c) in per.items()}
