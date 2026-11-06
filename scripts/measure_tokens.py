"""Tokens per word and Malayalam/English token ratio on FLORES-200 dev+devtest (parallel)."""
import sys
from pathlib import Path

import sentencepiece as spm
import tiktoken
from tokenizers import Tokenizer

flores, toks = Path(sys.argv[1]), Path(sys.argv[2])


def read(lang):
    return [l.strip() for split, ext in (("dev", "dev"), ("devtest", "devtest"))
            for l in (flores / split / f"{lang}.{ext}").read_text(encoding="utf-8").splitlines() if l.strip()]


en, ml = read("eng_Latn"), read("mal_Mlym")
assert len(en) == len(ml)

print("tiktoken encoding for gpt-5:", tiktoken.encoding_for_model("gpt-5").name)
print("tiktoken encoding for gpt-4:", tiktoken.encoding_for_model("gpt-4").name)

sp = spm.SentencePieceProcessor(model_file=str(toks / "pocket_multilingual.model"))
tokenizers = {
    "GPT-5 / GPT-4o (o200k_base)": lambda s: tiktoken.get_encoding("o200k_base").encode(s),
    "GPT-4 (cl100k_base)": lambda s: tiktoken.get_encoding("cl100k_base").encode(s),
    "SMC Malayalam Unigram 16k": lambda s: Tokenizer.from_file(str(toks / "malayalam-unigram-tokenizer.json")).encode(s).ids,
    "SMC Malayalam BPE 16k": lambda s: Tokenizer.from_file(str(toks / "malayalam-bpe-tokenizer.json")).encode(s).ids,
    "Pocket TTS multilingual SentencePiece 4k": lambda s: sp.encode(s),
}
# build each tokenizer once
built = {}
for name, f in tokenizers.items():
    if "o200k" in name:
        e = tiktoken.get_encoding("o200k_base"); built[name] = e.encode
    elif "cl100k" in name:
        e = tiktoken.get_encoding("cl100k_base"); built[name] = e.encode
    elif "Unigram" in name:
        t = Tokenizer.from_file(str(toks / "malayalam-unigram-tokenizer.json")); built[name] = lambda s, t=t: t.encode(s, add_special_tokens=False).ids
    elif "BPE" in name:
        t = Tokenizer.from_file(str(toks / "malayalam-bpe-tokenizer.json")); built[name] = lambda s, t=t: t.encode(s, add_special_tokens=False).ids
    else:
        built[name] = sp.encode

en_words = sum(len(s.split()) for s in en)
ml_words = sum(len(s.split()) for s in ml)
print(f"\n{len(en)} parallel sentences; words: English {en_words:,}, Malayalam {ml_words:,} "
      f"(Malayalam uses {ml_words / en_words:.2f}x as many words)\n")
print("| tokenizer | English tok/word | Malayalam tok/word | Malayalam tokens / English tokens (same text) |")
print("|---|---|---|---|")
for name, enc in built.items():
    ten = sum(len(enc(s)) for s in en)
    tml = sum(len(enc(s)) for s in ml)
    en_col = f"{ten / en_words:.2f}" if "SMC" not in name else "— (Malayalam-only)"
    ratio = f"{tml / ten:.2f}x" if "SMC" not in name else "—"
    print(f"| {name} | {en_col} | {tml / ml_words:.2f} | {ratio} |")

print("\nExample words:")
o200 = tiktoken.get_encoding("o200k_base")
uni = Tokenizer.from_file(str(toks / "malayalam-unigram-tokenizer.json"))
for w in ["കേരളത്തിന്റെ", "മരത്തിൽ", "ശാസ്ത്രജ്ഞന്മാർ", "കണ്ടുപിടിച്ചതായി", "ഉപയോഗിച്ച്", "computer", "scientists"]:
    pieces_uni = [p.replace("▁", "") for p in uni.encode(w, add_special_tokens=False).tokens]
    pieces_sp = [p.replace("▁", "") for p in sp.encode(w, out_type=str)]
    print(f"  {w}: o200k={len(o200.encode(w))} | SMC-Unigram={len(pieces_uni)} {pieces_uni} | Pocket-4k={len(pieces_sp)} {pieces_sp}")
