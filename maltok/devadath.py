"""Scoring against the sandhi gold data of Devadath et al. (github.com/Devadath/MLP_Word_Segementation, GPL-3.0;
see also Devadath V V, Kurisinkel, Sharma and Varma, "A Sandhi Splitter for Malayalam", ICON 2014).

The data is downloaded at run time, not redistributed. Each line is a word as space-separated characters and
one SP/NSP label per character; the split falls after an SP character (അതു|കൊണ്ട|ാണ്). It marks external
sandhi only (word + word). Their convention puts an inserted glide on the left (തുകയ|ാണ്); ours puts it on the
right (തുക|യാണ്), so both positions count, as does the start of the akshara containing the split.

Only recall is meaningful for a tokenizer here (tokenizers also cut at stem|suffix boundaries, which this data
does not mark): the share of gold split points where the tokenizer cuts, and the share of words where it cuts
at every gold split point.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

from .text import akshara_starts

BASE = "https://raw.githubusercontent.com/Devadath/MLP_Word_Segementation/master/data/"
FILES = {"gold 13.6k": "13.6k_gold_sandhi_char_level_training_data", "validation": "validation_data"}


def download(folder: Path) -> dict[str, Path]:
    folder.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, f in FILES.items():
        path = folder / f
        if not path.exists():
            urllib.request.urlretrieve(BASE + f, path)
        out[name] = path
    return out


def load(path: Path) -> list[tuple[str, list[int]]]:
    items = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 2:
            continue
        chars, labels = parts[0].split(" "), parts[1].split(" ")
        if len(chars) != len(labels):
            continue
        word = "".join(chars)
        cuts = [i + 1 for i, x in enumerate(labels) if x == "SP" and i + 1 < len(word)]
        if cuts:
            items.append((word, cuts))
    return items


def hit(word: str, gold_cut: int, cuts: set[int]) -> bool:
    starts = akshara_starts(word)
    ok = {gold_cut, max(s for s in starts if s <= gold_cut)}
    if word[gold_cut - 1] in "യവ":
        ok.add(gold_cut - 1)            # glide on the right (our convention)
    return bool(ok & cuts)


def score(cut_fn, items) -> dict:
    """cut_fn(word) -> set of cut positions, or None if it cannot handle the word (skipped)."""
    points = found = words = all_found = skipped = 0
    for word, gold in items:
        cuts = cut_fn(word)
        if cuts is None:
            skipped += 1
            continue
        words += 1
        h = [hit(word, g, cuts) for g in gold]
        points += len(h)
        found += sum(h)
        all_found += all(h)
    return {"point_recall": found / max(1, points), "word_recall": all_found / max(1, words),
            "words": words, "points": points, "skipped": skipped}
