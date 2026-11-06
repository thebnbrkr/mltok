"""Evaluate tokenizer pipelines on FLORES-200 Malayalam (dev + devtest).

    python -m maltok.evaluate --flores data/flores200_dataset --smc data/malayalam-unigram-tokenizer.json \
        --models results/tokenizers --out results/eval

Measures, per pipeline:
  tok/word        tokens per whitespace word on the raw sentences (each pipeline prepares text its own way)
  lone ▁          share of tokens that are only the word-start marker
  unk             share of <unk> tokens
  rényi eff.      Rényi efficiency (alpha 2.5) of the token distribution on FLORES (Zouhar et al. 2023)
  rule 1          on mlmorph-analysed nouns and verbs: word kept whole; among split words, "clean" =
                  a cut at the stem boundary (exact or akshara-rounded) and no cut inside the stem
  rule 2          share of cuts that fall inside an akshara
  robustness      share of words containing a chillu or ന്റ whose 2008-era spelling is cut the same way
                  (pieces compared after normalising each piece)
Rule 1 for the morph variant is partly circular: it was trained to respect the same mlmorph boundaries.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from . import morph
from .pipeline import Pipeline
from .text import MALAYALAM_RUN, NTA, OLD_CHILLU, VIRAMA, ZWNJ, akshara_starts, normalize, old_encoding

WS = "▁"


def flores_lines(flores: Path, lang: str = "mal_Mlym") -> list[str]:
    return [l.strip() for s in ("dev", "devtest")
            for l in (flores / s / f"{lang}.{s}").read_text(encoding="utf-8").splitlines() if l.strip()]


def word_cuts(pipe: Pipeline, word: str) -> tuple[set[int], bool]:
    """Internal cut positions in `word` (already normalised); False if the pieces do not spell it."""
    pieces = [p.replace(WS, "") for p in pipe.pieces(word)]
    pieces = [p for p in pieces if p]
    cuts, pos = set(), 0
    for p in pieces[:-1]:
        pos += len(p)
        cuts.add(pos)
    return cuts, "".join(pieces) == word


ZWSP = "\u200B"
VOWEL_SIGNS = set("\u0D3E\u0D3F\u0D40\u0D41\u0D42\u0D43\u0D46\u0D47\u0D48\u0D4A\u0D4B\u0D4C\u0D57")
CONSONANTS = {chr(c) for c in range(0x0D15, 0x0D3B)}


def _rng(word: str):
    import random
    import zlib
    return random.Random(zlib.crc32(word.encode()))      # same perturbation for every tokenizer


def with_zwsp(word: str) -> str | None:
    """An invisible zero-width space at a random akshara boundary inside the word."""
    cuts = sorted(c for c in akshara_starts(word) if 0 < c < len(word))
    if not cuts:
        return None
    i = _rng(word).choice(cuts)
    return word[:i] + ZWSP + word[i:]


def with_zwnj(word: str) -> str | None:
    """A zero-width non-joiner after a virama inside a conjunct (ക്ക -> ക്‌ക)."""
    spots = [i + 1 for i in range(len(word) - 1) if word[i] == VIRAMA and word[i + 1] in CONSONANTS]
    if not spots:
        return None
    i = _rng(word).choice(spots)
    return word[:i] + ZWNJ + word[i:]


def with_typo(word: str) -> str | None:
    """One vowel sign deleted."""
    spots = [i for i, ch in enumerate(word) if ch in VOWEL_SIGNS]
    if not spots:
        return None
    i = _rng(word).choice(spots)
    return word[:i] + word[i + 1:]


def evaluate(pipe: Pipeline, lines: list[str], words: list[str]) -> dict:
    n_words = sum(len(l.split()) for l in lines)
    toks = [t for l in lines for t in pipe.pieces(l)]
    ids = [i for l in lines for i in pipe.ids(l)]
    counts = Counter(ids)
    total = sum(counts.values())
    alpha = 2.5
    renyi = 1 / (1 - alpha) * math.log(sum((c / total) ** alpha for c in counts.values()))
    vocab = pipe.tok.get_vocab_size()
    r = {"name": pipe.name, "variant": pipe.variant, "vocab": vocab,
         "tok_per_word": len(toks) / n_words,
         "lone_ws": sum(t == WS for t in toks) / len(toks),
         "unk": sum(i == pipe.unk_id() for i in ids) / len(ids),
         "renyi_eff": renyi / math.log(vocab)}
    # rule 1 and rule 2 on distinct normalised words
    stats = {pos: Counter() for pos in ("noun", "verb")}
    cut_in = cut_all = mismatch = 0
    cmp_ = Counter()
    for w in words:
        cuts, ok = word_cuts(pipe, w)
        if not ok:
            mismatch += 1
            continue
        starts = akshara_starts(w)
        cut_all += len(cuts)
        cut_in += sum(c not in starts for c in cuts)
        comp = morph.compound(w)
        if comp:
            first = comp[0][0]
            cmp_["n"] += 1
            if cuts:
                cmp_["split"] += 1
                cmp_["clean"] += (first[0] in cuts or first[1] in cuts) and not any(0 < c < min(first) for c in cuts)
            else:
                cmp_["whole"] += 1
        b = morph.boundary(w)
        if b is None:
            continue
        pos, exact, rounded, _, _ = b
        s = stats[pos]
        s["n"] += 1
        if not cuts:
            s["whole"] += 1
            continue
        s["split"] += 1
        inner = any(0 < c < min(exact, rounded) for c in cuts)
        s["stem_cut"] += inner
        s["clean"] += (exact in cuts or rounded in cuts) and not inner
    r["mismatch_words"] = mismatch
    r["compound_n"] = cmp_["n"]
    r["compound_whole"] = cmp_["whole"] / max(1, cmp_["n"])
    r["compound_clean"] = cmp_["clean"] / max(1, cmp_["split"])
    r["akshara_cuts"] = cut_in / max(1, cut_all)
    for pos, s in stats.items():
        r[f"{pos}_n"] = s["n"]
        r[f"{pos}_whole"] = s["whole"] / max(1, s["n"])
        r[f"{pos}_clean"] = s["clean"] / max(1, s["split"])
        r[f"{pos}_stem_cut"] = s["stem_cut"] / max(1, s["split"])
    # robustness to 2008-era spellings
    affected = [w for w in words if NTA in w or any(ch in w for ch in OLD_CHILLU.values())]
    def split(word):   # pieces normalised one by one: "is the word cut the same way?", not "same bytes?"
        return [q for q in (normalize(p.replace(WS, "")) for p in pipe.pieces(word)) if q]
    same = sum(split(old_encoding(w)) == split(w) for w in affected)
    r["robust"] = same / max(1, len(affected))
    r["robust_n"] = len(affected)
    # harder perturbations no lookup table fixes: is the word still cut the same way (ignoring the inserted mark)?
    for key, perturb, mark in (("zwsp", with_zwsp, ZWSP), ("zwnj", with_zwnj, ZWNJ)):
        pairs = [(w, perturb(w)) for w in words]
        pairs = [(w, v) for w, v in pairs if v]
        ok = sum([q for q in (x.replace(mark, "") for x in split(v)) if q] == split(w) for w, v in pairs)
        r[f"robust_{key}"] = ok / max(1, len(pairs))
        r[f"robust_{key}_n"] = len(pairs)
    extra = [len(split(t)) - len(split(w)) for w in words for t in [with_typo(w)] if t]
    r["typo_extra"] = sum(extra) / max(1, len(extra))
    r["typo_frag"] = sum(e >= 2 for e in extra) / max(1, len(extra))
    r["typo_n"] = len(extra)
    return r


MAIN = [("tok_per_word", "tok/word", "{:.2f}"), ("lone_ws", "lone ▁", "{:.1%}"), ("unk", "unk", "{:.2%}"),
        ("renyi_eff", "Rényi eff.", "{:.3f}"), ("noun_whole", "nouns whole", "{:.1%}"),
        ("noun_clean", "nouns clean", "{:.1%}"), ("noun_stem_cut", "noun stem cut", "{:.1%}"),
        ("verb_whole", "verbs whole", "{:.1%}"), ("verb_clean", "verbs clean", "{:.1%}"),
        ("compound_whole", "compounds whole", "{:.1%}"), ("compound_clean", "compounds clean", "{:.1%}"),
        ("akshara_cuts", "cuts in akshara", "{:.1%}")]
ROBUST = [("robust", "2008 spelling: same cut", "{:.1%}"), ("robust_zwsp", "stray ZWSP: same cut", "{:.1%}"),
          ("robust_zwnj", "ZWNJ in conjunct: same cut", "{:.1%}"), ("typo_extra", "typo: extra tokens", "{:+.2f}"),
          ("typo_frag", "typo: +2 or more tokens", "{:.1%}")]


def table(results: list[dict], columns=MAIN) -> str:
    rows = ["| tokenizer | " + " | ".join(c[1] for c in columns) + " |", "|---|" + "---|" * len(columns)]
    for r in results:
        rows.append(f"| {r['name']} | " + " | ".join(f.format(r[k]) for k, _, f in columns) + " |")
    return "\n".join(rows)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flores", required=True)
    ap.add_argument("--wiki", help="ml.sentences.parquet: also evaluate on sentences from held-out articles")
    ap.add_argument("--wiki-n", type=int, default=5000)
    ap.add_argument("--smc", help="released SMC tokenizer.json, evaluated as a reference")
    ap.add_argument("--models", required=True, nargs="+", help="folder(s) with one sub-folder per trained pipeline")
    ap.add_argument("--out", required=True)
    ap.add_argument("--devadath", help="folder for Devadath et al.'s sandhi gold data (downloaded if missing)")
    a = ap.parse_args(argv)
    sets = {"FLORES-200 Malayalam dev+devtest (translated news-style text)": flores_lines(Path(a.flores))}
    if a.wiki:
        from .train import read_sentences
        sets[f"Malayalam Wikipedia, held-out articles ({a.wiki_n:,}-sentence sample; never used in training)"] = \
            read_sentences(Path(a.wiki), a.wiki_n, heldout=True)
    pipes = ([Pipeline.reference(Path(a.smc))] if a.smc else []) + \
            [Pipeline.load(p) for m in a.models for p in sorted(Path(m).iterdir()) if (p / "meta.json").exists()]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    md, everything = [], {}
    for title, lines in sets.items():
        words = sorted({w for l in lines for w in MALAYALAM_RUN.findall(normalize(l)) if len(w) > 1})
        morph.precompute(words, with_fallback=any(p.variant == "compound" for p in pipes))
        results = [evaluate(p, lines, words) for p in pipes]
        n = results[0]
        md += [f"### {title}", "",
               f"{len(lines):,} sentences, {len(words):,} distinct words; mlmorph-analysed stems: "
               f"{n['noun_n']:,} nouns, {n['verb_n']:,} verbs, {n['compound_n']:,} compounds; {n['robust_n']:,} words with a chillu or ന്റ "
               "for the 2008-spelling test.", "", "**Compactness and morphology**", "", table(results), "",
               f"**Robustness** (stray ZWSP: {n['robust_zwsp_n']:,} words; ZWNJ: {n['robust_zwnj_n']:,} words "
               f"with a conjunct; typo, one vowel sign deleted: {n['typo_n']:,} words)", "",
               table(results, ROBUST), ""]
        everything[title] = results
    from . import gold
    if gold.GOLD.exists():
        rows = gold.load()
        kinds = sorted({r["sandhi"] for r in rows})
        res = {p.name: gold.evaluate(p, rows) for p in pipes}
        head = "| tokenizer | all: join found | all: clean | " + " | ".join(f"{k}: found" for k in kinds) + " |"
        md += ["### പിരിച്ചെഴുതുക gold set (draft; to be checked by a native speaker)", "",
               f"{len(rows)} joined words with their school split. found = the tokenizer cuts at every join; "
               "clean = and does not cut inside the first part.", "", head, "|---|" + "---|" * (2 + len(kinds))]
        for name, g in res.items():
            cells = [f"{g['all']['found']:.0%} ({g['all']['n']})", f"{g['all']['clean']:.0%}"] + \
                    [f"{g[k]['found']:.0%} ({g[k]['n']})" if k in g else "—" for k in kinds]
            md.append(f"| {name} | " + " | ".join(cells) + " |")
        md.append("")
        everything["gold"] = res
    if a.devadath:
        from . import devadath
        files = devadath.download(Path(a.devadath))
        md += ["### Devadath et al. sandhi gold data (external sandhi, word + word)", "",
               "Recall only: share of gold split points where the cutter cuts (point), and of words with every "
               "gold point cut (word). Both glide conventions count. Words a tokenizer re-spells (normalisation) "
               "are skipped.", ""]
        cutters = [(p.name, (lambda p: lambda w: (lambda c: c[0] if c[1] else None)(word_cuts(p, w)))(p))
                   for p in pipes]
        cutters += [("splitter: mlmorph compounds", lambda w: set((morph.compound(w) or ([], []))[1])),
                    ("splitter: mlmorph + fallback", lambda w: set((morph.compound(w, True) or ([], []))[1]))]
        dv = {}
        for name, path in files.items():
            items = devadath.load(path)
            morph.precompute([w for w, _ in items], with_fallback=True)
            dv[name] = {n: devadath.score(fn, items) for n, fn in cutters}
        md += ["| cutter | " + " | ".join(f"{k}: point · word" for k in dv) + " |", "|---|" + "---|" * len(dv)]
        for n, _ in cutters:
            md.append(f"| {n} | " + " | ".join(f"{dv[k][n]['point_recall']:.1%} · {dv[k][n]['word_recall']:.1%} "
                                                f"({dv[k][n]['words']:,})" for k in dv) + " |")
        md.append("")
        everything["devadath"] = dv
    (out / "eval.json").write_text(json.dumps(everything, indent=1, ensure_ascii=False))
    (out / "eval.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
