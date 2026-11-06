"""Check Malayalam tokenizers against two grammar rules, on FLORES-200 (dev+devtest) Malayalam words.

Rule 1, morphology (Keralapaniniyam cases + plural): for nouns that mlmorph analyses as
  stem<n|np>[<pl>][<case>], the silver stem boundary is the longest common prefix of the surface
  word and its lemma, moved back to the start of an akshara (മരം/മരത്തിൽ -> മര|ത്തിൽ,
  വീട്/വീട്ടിൽ -> വീ|ട്ടിൽ). MorphScore-style: does the tokenizer cut exactly there?
Rule 2, script: a token boundary should not fall inside an akshara (consonant cluster + vowel sign/virama).
"""
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import sentencepiece as spm
import tiktoken
from mlmorph import Analyser
from tokenizers import Tokenizer

flores, toks = Path(sys.argv[1]), Path(sys.argv[2])
lines = [l for s in ("dev", "devtest") for l in (flores / s / f"mal_Mlym.{s}").read_text(encoding="utf-8").splitlines()]
words = Counter(w for l in lines for w in re.findall(r"[ഀ-ൿ‌‍]+", l))
words = Counter({w: c for w, c in words.items() if len(w) > 1})

C = "[ക-ഹഺ]"
AKSHARA = re.compile(rf"(?:{C}(?:്{C})*(?:[ാ-ൌൗൢൣ]|്[‌‍]?)?[ംഃ]?)"
                     r"|(?:[അ-ഔൠൡ][ംഃ]?)|[ൺ-ൿൔ-ൖ]|.", re.S)


def akshara_starts(w):
    return {m.start() for m in AKSHARA.finditer(w)} | {len(w)}


def round_down(pos, starts):
    return max(s for s in starts if s <= pos)


o200 = tiktoken.get_encoding("o200k_base")
sp = spm.SentencePieceProcessor(model_file=str(toks / "pocket_multilingual.model"))
uni = Tokenizer.from_file(str(toks / "malayalam-unigram-tokenizer.json"))
bpe = Tokenizer.from_file(str(toks / "malayalam-bpe-tokenizer.json"))


def cuts_from_pieces(w, pieces):
    """Internal character boundaries from a list of piece strings that spell w (word-start marker removed)."""
    out, pos = set(), 0
    for p in pieces:
        p = p.replace("▁", "")
        if not p:
            continue
        pos += len(p)
        out.add(pos)
    return {c for c in out if 0 < c < len(w)}, "".join(p.replace("▁", "") for p in pieces) == w


def cuts_o200(w):
    """Character boundaries for byte-level o200k; a boundary inside a character's bytes is reported as 'mid-char'."""
    ids = o200.encode(w)
    b, out, mid = 0, set(), 0
    char_end_bytes = {}
    acc = 0
    for i, ch in enumerate(w):
        acc += len(ch.encode("utf-8"))
        char_end_bytes[acc] = i + 1
    for t in ids[:-1]:
        b += len(o200.decode_single_token_bytes(t))
        if b in char_end_bytes:
            out.add(char_end_bytes[b])
        else:
            mid += 1
    return out, mid


TOKENIZERS = {
    "SMC Unigram 16k": lambda w: cuts_from_pieces(w, uni.encode(w, add_special_tokens=False).tokens),
    "SMC BPE 16k": lambda w: cuts_from_pieces(w, bpe.encode(w, add_special_tokens=False).tokens),
    "Pocket multilingual 4k": lambda w: cuts_from_pieces(w, sp.encode(w, out_type=str)),
}

an = Analyser()
CASES = ("locative", "genitive", "dative", "accusative", "sociative", "instrumental")
nouns = {}
for w in words:
    res = an.analyse(w)
    if not res:
        continue
    m = re.fullmatch(r"([^<]+)<(n|np)>((?:<pl>)?)(?:<(" + "|".join(CASES) + r")>)?", res[0][0])
    if not m or (not m.group(3) and not m.group(4)):
        continue
    lemma = m.group(1)
    lcp = 0
    while lcp < min(len(w), len(lemma)) and w[lcp] == lemma[lcp]:
        lcp += 1
    b = round_down(lcp, akshara_starts(w))
    if 0 < b < len(w):
        nouns[w] = (b, lcp, (m.group(4) or "plural-only") + ("+pl" if m.group(3) and m.group(4) else ""))

analysed = sum(1 for w in words if an.analyse(w))
print(f"FLORES Malayalam: {len(words):,} distinct words; mlmorph analyses {analysed:,} ({analysed / len(words):.0%}); "
      f"{len(nouns):,} inflected nouns used for rule 1")
print("cases among them:", dict(Counter(t for *_, t in nouns.values()).most_common()))

print("\nRule 1 on the inflected nouns: whole = kept as one token; among split words: at-syllable = cut at the "
      "akshara-rounded stem boundary, at-morpheme = cut at the exact stem end (may fall inside an akshara), "
      "clean = cut at either AND the stem itself not split. Rule 2 on all distinct words.\n")
print("| tokenizer | tokens/word | whole word | at-syllable | at-morpheme | clean stem+suffix | stem split | rule 2: cuts inside an akshara |")
print("|---|---|---|---|---|---|---|---|")
for name, f in list(TOKENIZERS.items()) + [("GPT-5 o200k", None)]:
    cut_in = cut_all = mid = 0
    whole = n_split = at_syl = at_morph = clean = stem_split = 0
    for w in words:
        if f is None:
            cuts, m = cuts_o200(w); mid += m
        else:
            cuts, _ = f(w)
        starts = akshara_starts(w)
        cut_all += len(cuts) + (m if f is None else 0)
        cut_in += sum(1 for c in cuts if c not in starts) + (m if f is None else 0)
        if w in nouns:
            b, e, _ = nouns[w]
            if not cuts and not (f is None and m):
                whole += 1; continue
            n_split += 1
            hs, hm = b in cuts, e in cuts
            at_syl += hs; at_morph += hm
            inner = any(0 < c < min(b, e) for c in cuts)
            stem_split += inner
            clean += (hs or hm) and not inner
    ntok = sum(len(o200.encode(w)) for w in words) if f is None else sum(len(f(w)[0]) + 1 for w in words)
    pct = lambda x, d: f"{100 * x / max(1, d):.1f}%"
    extra = f" ({mid:,} split one character's bytes)" if f is None else ""
    print(f"| {name} | {ntok / len(words):.2f} | {pct(whole, len(nouns))} | {pct(at_syl, n_split)} | {pct(at_morph, n_split)} | "
          f"{pct(clean, n_split)} | {pct(stem_split, n_split)} | {pct(cut_in, cut_all)}{extra} |")

print("\nExamples (silver stem boundary shown with |, then each tokenizer's pieces):")
show = ["മരത്തിൽ", "രാജ്യങ്ങളുടെ", "കുട്ടികൾക്ക്", "വീട്ടിൽ", "സർക്കാരിന്റെ", "ആശുപത്രിയിൽ", "ജനങ്ങൾ", "നഗരത്തിൽ"]
show = [w for w in show if w in nouns] + [w for w in list(nouns)[:200] if len(w) > 7][:6]


def split_at(w, cuts):
    out, prev = [], 0
    for c in sorted(cuts) + [len(w)]:
        out.append(w[prev:c]); prev = c
    return " · ".join(out)


for w in show[:12]:
    b, e, tag = nouns[w]
    row = [f"{w[:b]}|{w[b:]} ({tag})"]
    for name, f in TOKENIZERS.items():
        row.append(f"{name.split()[0]} {name.split()[1]}: {split_at(w, f(w)[0])}")
    row.append(f"o200k: {split_at(w, cuts_o200(w)[0])}")
    print("  " + "  ||  ".join(row))
