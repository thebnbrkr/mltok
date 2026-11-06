"""What is inside the pocket-tts-multilingual 4k SentencePiece: pieces per script, fertility per language."""
import sys, unicodedata
from collections import Counter
from pathlib import Path
import sentencepiece as spm
model, flores = sys.argv[1], Path(sys.argv[2])
sp = spm.SentencePieceProcessor(model_file=model)
def script(piece):
    for ch in piece.replace("▁", ""):
        n = unicodedata.name(ch, "")
        for s in ("MALAYALAM", "TAMIL", "TELUGU", "KANNADA", "DEVANAGARI", "LATIN"):
            if n.startswith(s): return s
        if ch.isdigit(): return "DIGIT"
    return "other/space"
c = Counter(script(sp.id_to_piece(i)) for i in range(sp.get_piece_size()))
print("pieces:", sp.get_piece_size(), dict(c.most_common()))
print("model type:", "unigram" if sp.get_score(5) != 0 else "bpe?", "| example Malayalam pieces:",
      [sp.id_to_piece(i) for i in range(sp.get_piece_size()) if script(sp.id_to_piece(i)) == "MALAYALAM"][:25])
langs = {"eng_Latn": "English", "hin_Deva": "Hindi", "mar_Deva": "Marathi", "kan_Knda": "Kannada",
         "tam_Taml": "Tamil", "tel_Telu": "Telugu", "mal_Mlym": "Malayalam"}
wer = {"English": 6.2, "Hindi": 10.5, "Marathi": 5.8, "Tamil": 11.0, "Telugu": 13.8, "Kannada": 14.7, "Malayalam": 23.7}
print("\n| language | pieces in vocab | tokens/word (FLORES) | chars/token | model card WER % |\n|---|---|---|---|---|")
scr = {"English": "LATIN", "Hindi": "DEVANAGARI", "Marathi": "DEVANAGARI", "Kannada": "KANNADA", "Tamil": "TAMIL", "Telugu": "TELUGU", "Malayalam": "MALAYALAM"}
for code, name in langs.items():
    lines = [l for s in ("dev", "devtest") for l in (flores / s / f"{code}.{s}").read_text(encoding="utf-8").splitlines() if l.strip()]
    toks = sum(len(sp.encode(l)) for l in lines); words = sum(len(l.split()) for l in lines); chars = sum(len(l.replace(" ", "")) for l in lines)
    print(f"| {name} | {c[scr[name]]} | {toks / words:.2f} | {chars / toks:.2f} | {wer[name]} |")
