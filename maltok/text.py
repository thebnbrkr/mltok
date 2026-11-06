"""Normalisation and akshara (orthographic syllable) rules for Malayalam.

normalize() maps encodings that differ only in how they were typed onto one form:
  - NFC;
  - the three spellings of ന്റ (2008-era ZWJ form ന്‍റ, chillu form ൻ്റ, and ന്റ) -> ന്റ (na, virama, rra);
  - 2008-era chillus (consonant + virama + ZWJ, before Unicode 5.1 added atomic chillus) -> atomic chillus.
Archaic letters are left alone.
"""
from __future__ import annotations

import re
import unicodedata

VIRAMA, ZWJ, ZWNJ = "്", "‍", "‌"
NTA = "ന്റ"                       # ന്റ
NTA_VARIANTS = ("ന്‍റ",       # ന്‍റ  (2008-era)
                "ൻ്റ")             # ൻ്റ   (chillu n + virama + rra)
OLD_CHILLU = {"ണ": "ൺ", "ന": "ൻ", "ര": "ർ",   # ണ ന ര
              "ല": "ൽ", "ള": "ൾ", "ക": "ൿ"}   # ല ള ക
MALAYALAM_RUN = re.compile(r"[ഀ-ൿ‌‍]+")

_C = "[ക-ഺ]"
AKSHARA = re.compile(
    rf"(?:{_C}(?:{VIRAMA}{_C})*(?:[ാ-ൌൗൢൣ]|{VIRAMA}[{ZWNJ}{ZWJ}]?)?[ംഃ]?)"
    r"|(?:[അ-ഔൠൡ][ംഃ]?)|[ൺ-ൿൔ-ൖ]|.", re.S)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    for v in NTA_VARIANTS:                        # before chillus: ന്‍റ contains ന്‍
        text = text.replace(v, NTA)
    for consonant, chillu in OLD_CHILLU.items():
        text = text.replace(consonant + VIRAMA + ZWJ, chillu)
    return text


def aksharas(word: str) -> list[str]:
    return [m.group() for m in AKSHARA.finditer(word)]


def akshara_starts(word: str) -> set[int]:
    """Character offsets where an akshara starts, plus len(word)."""
    return {m.start() for m in AKSHARA.finditer(word)} | {len(word)}


def old_encoding(word: str) -> str:
    """The 2008-era spelling of a normalised word (inverse of normalize), for robustness tests."""
    word = word.replace(NTA, NTA_VARIANTS[0])
    for consonant, chillu in OLD_CHILLU.items():
        word = word.replace(chillu, consonant + VIRAMA + ZWJ)
    return word
