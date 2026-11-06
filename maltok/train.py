"""Train tokenizer variants on Malayalam Wikipedia sentences (smcproject/ml-wiki-sentences).

    python -m maltok.train --data data/ml.sentences.parquet --variants base norm morph akshara \
        --vocab 16000 --out results/tokenizers [--max-sentences 200000]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .pipeline import VARIANTS, train


HELDOUT_MOD = 100   # articles whose URL hashes to 0 mod 100 (~1%) are held out for evaluation


def is_heldout(url: str) -> bool:
    import zlib
    return zlib.crc32((url or "").encode()) % HELDOUT_MOD == 0


def read_sentences(path: Path, limit: int | None = None, seed: int = 0, heldout: bool = False) -> list[str]:
    """Training sentences (heldout=False) or held-out-article sentences (heldout=True), split by article."""
    import pyarrow.parquet as pq
    t = pq.read_table(path, columns=["sentence", "url"])
    lines = [s for s, u in zip(t.column("sentence").to_pylist(), t.column("url").to_pylist())
             if s and s.strip() and is_heldout(u) == heldout]
    if limit and limit < len(lines):
        import random
        lines = random.Random(seed).sample(lines, limit)
    return lines


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    ap.add_argument("--vocab", type=int, default=16000)
    ap.add_argument("--max-sentences", type=int)
    ap.add_argument("--char-coverage", type=float, help="e.g. 0.9999: drop the rarest characters from training "
                    "(SentencePiece-style); omit to keep every character (the SMC recipe)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-existing", action="store_true")
    a = ap.parse_args(argv)
    lines = read_sentences(Path(a.data), a.max_sentences)
    print(f"{len(lines):,} training sentences", flush=True)
    for v in a.variants:
        name = f"{v}-{a.vocab // 1000}k" + (f"-{a.max_sentences // 1000}ks" if a.max_sentences else "") + \
            (f"-cov{a.char_coverage:g}" if a.char_coverage else "")
        out = Path(a.out) / name
        if a.skip_existing and (out / "meta.json").exists():
            print("skip", name)
            continue
        t0 = time.time()
        pipe = train(v, lines, a.vocab, a.char_coverage)
        pipe.name = name
        pipe.save(out)
        (out / "train.json").write_text(json.dumps({"sentences": len(lines), "vocab": a.vocab,
                                                    "char_coverage": a.char_coverage,
                                                    "seconds": time.time() - t0}))
        print(f"{name}: trained in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
