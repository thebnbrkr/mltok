"""Tokenizer variants and a common interface for them.

All variants use the SMC recipe (Thottingal's malayalam-unigram-tokenizer): HF `tokenizers` Unigram,
Strip + NFC normaliser, Metaspace pre-tokenizer (prepend always), the same five special tokens.
They differ only in what is done to the text first:

  base      nothing (the SMC recipe retrained on our data)
  norm      maltok.text.normalize (2008-era chillus, ന്റ spellings)
  morph     norm + a forced boundary at mlmorph's stem|suffix cut (pieces never cross it)
  akshara   norm + every akshara mapped to one private-use character, so pieces are whole aksharas
            (aksharas outside the map, i.e. rare ones, stay as characters)
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

from . import morph
from .text import MALAYALAM_RUN, aksharas, normalize

VARIANTS = ("base", "norm", "morph", "akshara")
SPECIAL = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
PUA_START, PUA_SIZE = 0xE000, 6400        # Basic Multilingual Plane private-use area


def new_tokenizer(variant: str) -> Tokenizer:
    tok = Tokenizer(models.Unigram())
    tok.normalizer = normalizers.Sequence([normalizers.Strip(), normalizers.NFC()])
    meta = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always", split=True)
    if variant == "morph":
        tok.pre_tokenizer = pre_tokenizers.Sequence([meta, pre_tokenizers.Split(morph.MARK, behavior="removed")])
    else:
        tok.pre_tokenizer = meta
    tok.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always", split=True)
    return tok


def akshara_map(lines, size: int = PUA_SIZE) -> dict[str, str]:
    """The `size` most frequent multi-character aksharas -> one private-use character each."""
    counts = Counter(a for line in lines for run in MALAYALAM_RUN.findall(line) for a in aksharas(run) if len(a) > 1)
    return {a: chr(PUA_START + i) for i, (a, _) in enumerate(counts.most_common(size))}


class Pipeline:
    """Text -> pieces (strings in the prepared text, word-start marker kept) for one variant."""

    def __init__(self, tokenizer: Tokenizer, variant: str, amap: dict | None = None, name: str | None = None):
        self.tok, self.variant, self.name = tokenizer, variant, name or variant
        self.amap = amap or {}
        self.inverse = {v: k for k, v in self.amap.items()}

    # ---- preparation shared by training and encoding
    def prepare(self, text: str) -> str:
        if self.variant != "base":
            text = normalize(text)
        if self.variant == "morph":
            text = morph.segment_text(text)
        if self.variant == "akshara":
            text = MALAYALAM_RUN.sub(lambda m: "".join(self.amap.get(a, a) for a in aksharas(m.group())), text)
        return text

    def pieces(self, text: str) -> list[str]:
        toks = self.tok.encode(self.prepare(text), add_special_tokens=False).tokens
        if self.inverse:
            toks = ["".join(self.inverse.get(c, c) for c in t) for t in toks]
        return toks

    def unk_id(self) -> int:
        return self.tok.token_to_id("<unk>")

    def ids(self, text: str) -> list[int]:
        return self.tok.encode(self.prepare(text), add_special_tokens=False).ids

    # ---- persistence
    def save(self, out: Path) -> None:
        out.mkdir(parents=True, exist_ok=True)
        self.tok.save(str(out / "tokenizer.json"))
        (out / "meta.json").write_text(json.dumps({"variant": self.variant, "name": self.name}))
        if self.amap:
            (out / "akshara_map.json").write_text(json.dumps(self.amap, ensure_ascii=False))

    @classmethod
    def load(cls, path: Path) -> "Pipeline":
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        amap = json.loads((path / "akshara_map.json").read_text()) if (path / "akshara_map.json").exists() else None
        return cls(Tokenizer.from_file(str(path / "tokenizer.json")), meta["variant"], amap, meta.get("name"))

    @classmethod
    def reference(cls, tokenizer_json: Path, name: str = "smc-released") -> "Pipeline":
        """A released tokenizer used as is (no preparation)."""
        return cls(Tokenizer.from_file(str(tokenizer_json)), "base", None, name)


def train(variant: str, lines: list[str], vocab_size: int = 16000) -> Pipeline:
    assert variant in VARIANTS, variant
    amap = akshara_map(lines) if variant == "akshara" else None
    pipe = Pipeline(new_tokenizer(variant), variant, amap)
    if variant == "morph":
        words = (run for line in lines for run in MALAYALAM_RUN.findall(normalize(line)))
        morph.precompute(words)
    trainer = trainers.UnigramTrainer(vocab_size=vocab_size, special_tokens=SPECIAL, unk_token="<unk>")
    pipe.tok.train_from_iterator((pipe.prepare(l) for l in lines), trainer=trainer, length=len(lines))
    return pipe
