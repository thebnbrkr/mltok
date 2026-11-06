"""How tokenizers and mlmorph handle old letters and encoding variants of the same Malayalam text.

    python scripts/old_letters.py data
"""
import sys
from pathlib import Path

import sentencepiece as spm
from mlmorph import Analyser
from tokenizers import Tokenizer

t = Path(sys.argv[1])
uni = Tokenizer.from_file(str(t / "malayalam-unigram-tokenizer.json"))
sp = spm.SentencePieceProcessor(model_file=str(t / "pocket_multilingual.model"))
an = Analyser()
CASES = [
    ("dot reph (old)", "ധൎമ്മം"), ("chillu r (modern)", "ധർമ്മം"),
    ("dot reph (old)", "കൎമ്മം"), ("chillu r (modern)", "കർമ്മം"),
    ("nta: old ZWJ", "അവന്‍റെ"), ("nta: n+virama+rra", "അവന്റെ"), ("nta: chillu n+virama+rra", "അവൻ്റെ"),
    ("old chillu l (ZWJ)", "അവരില്‍"), ("atomic chillu l", "അവരിൽ"),
    ("Malayalam digits", "൧൯൯൬"), ("archaic letter nnna", "ഩ"), ("archaic letter ttta", "ഺ"),
    ("archaic ii", "ൟ"), ("date mark", "൹"),
]
print("| form | text | SMC Unigram pieces | Pocket 4k pieces | mlmorph |")
print("|---|---|---|---|---|")
for label, w in CASES:
    u = uni.encode(w, add_special_tokens=False)
    up = ["UNK" if i == uni.token_to_id("<unk>") else p for p, i in zip(u.tokens, u.ids)]
    pp = ["UNK" if i == sp.unk_id() else p for p, i in zip(sp.encode(w, out_type=str), sp.encode(w))]
    a = an.analyse(w)
    print(f"| {label} | {w!r} | {len(up)}: {' · '.join(up)} | {len(pp)}: {' · '.join(pp)} | {a[0][0] if a else '—'} |")
