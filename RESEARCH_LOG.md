# Research log: Malayalam tokenizer

A running diary of what we tried, what happened and what we changed. Entries are appended, never rewritten.
Separate from the grassman and Hyena-audio logs.

---

## 2026-11-06: Starting point, measured on FLORES-200 Malayalam (dev + devtest, 2,009 sentences)

**Goal:** a better Malayalam tokenizer than the SMC Unigram tokenizer (Thottingal, 2026), and then a test of
whether it helps a Malayalam Pocket TTS. Credit: the SMC tokenizer, its recipe and mlmorph are Thottingal's
(MIT); we build on them.

### Tokens per word (same 2,009 sentences in English and Malayalam)

| tokenizer | English | Malayalam | Malayalam ÷ English tokens |
|---|---|---|---|
| GPT-5 / GPT-4o (o200k_base) | 1.22 | 3.50 | 1.95× |
| GPT-4 (cl100k_base) | 1.24 | 16.16 | 8.94× |
| SMC Unigram 16k | — | 2.13 | — |
| SMC BPE 16k | — | 2.11 | — |
| Pocket TTS multilingual 4k (Thanush) | 2.59 | 4.28 | 1.13× |

The Pocket multilingual tokenizer gives Malayalam 601 of its 4,000 pieces, mostly sub-syllable fragments. Across its
7 languages, Malayalam has the most tokens per word (4.28) and the worst WER on its model card (23.7%); with
unknown data per language, data and tokenizer cannot be separated there.

### Grammar check (1,447 mlmorph-analysed inflected nouns; silver boundaries)

| tokenizer | clean stem\|suffix | stem itself cut | cuts inside an akshara |
|---|---|---|---|
| SMC Unigram 16k | 76.2% | 15.7% | 36.0% |
| SMC BPE 16k | 67.3% | 30.3% | 37.4% |
| Pocket multilingual 4k | 7.0% | 93.0% | 38.2% |
| GPT-5 o200k | 5.6% | 94.1% | 56.9% |

### Weaknesses of the SMC tokenizer found so far

1. **2008-era chillus are not normalised** (consonant + virama + ZWJ, before Unicode 5.1): 96 of 2,009 FLORES
   sentences; `അ · ര്‍ · ബു · ദ · ത്തിന്` instead of `അർബുദ · ത്തിന്`; tokens per word on those sentences
   2.33 → 2.20 after normalising.
2. **Three spellings of ന്റ** are tokenized differently; `അവൻ്റെ` → `അവ · ൻ · ്റെ` (a piece starting with a virama).
3. **Lone word-start token `▁`:** 4.2% of all its tokens, on 8.9% of words; without them 2.12 → 2.03 tokens per word.
4. **Morpheme vs syllable conflict:** many of its cuts inside an akshara are at morpheme boundaries whose suffix
   is a vowel sign (`തരംതിരിക്ക|ാൻ`, `സാധിക്ക|ുന്ന`).

### Plan (notebooks/tokenizer_colab.ipynb)

Four variants of the SMC recipe (Unigram 16k, Strip + NFC, Metaspace), trained on `smcproject/ml-wiki-sentences`
(2.25M sentences, CC-BY-SA), evaluated on FLORES with the released SMC tokenizer as reference:
`base` (recipe retrained), `norm` (normaliser), `morph` (norm + mlmorph boundaries), `akshara` (norm + whole
aksharas). Decision rule: a variant is **better** if it improves the morphology and robustness columns without
making tokens per word more than 5% worse.

### Predictions (before running)

- `base` vs `smc-released`: within ±0.1 tokens per word; Wikipedia vs the SMC corpus changes little.
- `norm` vs `base`: robustness to 2008 spellings ~100% (by construction); tokens per word on all of FLORES drops
  by < 0.02 (only 5% of sentences are affected); lone `▁` unchanged.
- `morph` vs `norm`: nouns clean stem|suffix > 90% (partly circular), tokens per word up 3–10% (the Hungarian
  Morfessor vocabulary cost 7–9%).
- `akshara`: cuts inside an akshara ~0% for mapped aksharas; nouns clean drops (vowel-sign suffixes can no
  longer be split off); tokens per word within 5% of `norm`.
- None of the variants removes the lone `▁` problem; that needs a separate fix.

### Limits

Silver boundaries (mlmorph lemma + akshara rounding), nouns and simple verbs only; one evaluation corpus; rule 1
for `morph` is partly circular. A native speaker will check ~300 words to make a gold set.

---

## 2026-11-06 (later): Second evaluation set, after Thottingal's MTEB post

His MTEB post (Oct 2026) shows Malayalam's leaderboard score is 84% one FLORES-based task, and argues for
evaluation on real Malayalam text. FLORES is translated news-style text, so we add a second set: sentences from
**held-out Wikipedia articles** (articles whose URL hashes to 0 mod 100, about 1%; split by article, so no article
is in both training and test; 5,000-sentence sample). Same measures, reported separately for both sets.

Compounds (his ആനപ്പുറത്തു കയറി example) are left for the native-speaker gold set: mlmorph's compound analyses
need a separate boundary rule.

---

## 2026-11-06 (later): Pilot on Colab: training works, evaluation crashed (fixed)

Pilot (200k sentences, 16k vocabulary) training times on Colab: base 110 s, norm 113 s, morph 190 s (mlmorph
analysis included), akshara 70 s.

**Bug:** the evaluation could not find the FLORES files. The archive's member names start with `./`
(`./flores200_dataset/dev/mal_Mlym.dev`). The Mac's bsdtar matched them without the `./`, Colab's GNU tar did not,
so nothing was extracted, and `curl -s` hid the failure. **Fix:** the data cell now extracts with Python's
`tarfile`, matching on the end of the name, and asserts that both files have the expected number of lines.
The trained pilot tokenizers are unaffected; re-running the stage with `--skip-existing` only evaluates.
