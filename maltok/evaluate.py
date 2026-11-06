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
  robustness      share of words containing a chillu or ന്റ whose 2008-era spelling gives the same pieces
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
from .text import MALAYALAM_RUN, NTA, OLD_CHILLU, akshara_starts, normalize, old_encoding

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
    for w in words:
        cuts, ok = word_cuts(pipe, w)
        if not ok:
            mismatch += 1
            continue
        starts = akshara_starts(w)
        cut_all += len(cuts)
        cut_in += sum(c not in starts for c in cuts)
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
    r["akshara_cuts"] = cut_in / max(1, cut_all)
    for pos, s in stats.items():
        r[f"{pos}_n"] = s["n"]
        r[f"{pos}_whole"] = s["whole"] / max(1, s["n"])
        r[f"{pos}_clean"] = s["clean"] / max(1, s["split"])
        r[f"{pos}_stem_cut"] = s["stem_cut"] / max(1, s["split"])
    # robustness to 2008-era spellings
    affected = [w for w in words if NTA in w or any(ch in w for ch in OLD_CHILLU.values())]
    same = sum([p.replace(WS, "") for p in pipe.pieces(old_encoding(w))] ==
               [p.replace(WS, "") for p in pipe.pieces(w)] for w in affected)
    r["robust"] = same / max(1, len(affected))
    r["robust_n"] = len(affected)
    return r


COLUMNS = [("tok_per_word", "tok/word", "{:.2f}"), ("lone_ws", "lone ▁", "{:.1%}"), ("unk", "unk", "{:.2%}"),
           ("renyi_eff", "Rényi eff.", "{:.3f}"), ("noun_whole", "nouns whole", "{:.1%}"),
           ("noun_clean", "nouns clean", "{:.1%}"), ("noun_stem_cut", "noun stem cut", "{:.1%}"),
           ("verb_whole", "verbs whole", "{:.1%}"), ("verb_clean", "verbs clean", "{:.1%}"),
           ("akshara_cuts", "cuts in akshara", "{:.1%}"), ("robust", "2008 spelling same", "{:.1%}")]


def table(results: list[dict]) -> str:
    head = "| tokenizer | " + " | ".join(c[1] for c in COLUMNS) + " |"
    rows = [head, "|---|" + "---|" * len(COLUMNS)]
    for r in results:
        rows.append(f"| {r['name']} | " + " | ".join(f.format(r[k]) for k, _, f in COLUMNS) + " |")
    return "\n".join(rows)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flores", required=True)
    ap.add_argument("--wiki", help="ml.sentences.parquet: also evaluate on sentences from held-out articles")
    ap.add_argument("--wiki-n", type=int, default=5000)
    ap.add_argument("--smc", help="released SMC tokenizer.json, evaluated as a reference")
    ap.add_argument("--models", required=True, help="folder with one sub-folder per trained pipeline")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    sets = {"FLORES-200 Malayalam dev+devtest (translated news-style text)": flores_lines(Path(a.flores))}
    if a.wiki:
        from .train import read_sentences
        sets[f"Malayalam Wikipedia, held-out articles ({a.wiki_n:,}-sentence sample; never used in training)"] = \
            read_sentences(Path(a.wiki), a.wiki_n, heldout=True)
    pipes = ([Pipeline.reference(Path(a.smc))] if a.smc else []) + \
            [Pipeline.load(p) for p in sorted(Path(a.models).iterdir()) if (p / "meta.json").exists()]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    md, everything = [], {}
    for title, lines in sets.items():
        words = sorted({w for l in lines for w in MALAYALAM_RUN.findall(normalize(l)) if len(w) > 1})
        morph.precompute(words)
        results = [evaluate(p, lines, words) for p in pipes]
        n = results[0]
        md += [f"### {title}", "",
               f"{len(lines):,} sentences, {len(words):,} distinct words; mlmorph-analysed stems: "
               f"{n['noun_n']:,} nouns, {n['verb_n']:,} verbs; {n['robust_n']:,} words with a chillu or ന്റ "
               "for the robustness test.", "", table(results), ""]
        everything[title] = results
    (out / "eval.json").write_text(json.dumps(everything, indent=1, ensure_ascii=False))
    (out / "eval.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
