# malayalam-tok

Checks of Malayalam tokenizers against Malayalam grammar, as groundwork for a tokenizer × TTS study
(Pocket TTS; Hungarian counterpart in the Pocket-hu / IWCLUL 2026 work).

## Setup

```bash
pip install -r requirements.txt
mkdir -p data && cd data
curl -L -O https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz && tar -xzf flores200_dataset.tar.gz
curl -L -o malayalam-unigram-tokenizer.json https://huggingface.co/smcproject/malayalam-unigram-tokenizer/resolve/main/tokenizer.json
curl -L -o malayalam-bpe-tokenizer.json https://huggingface.co/smcproject/malayalam-bpe-tokenizer/resolve/main/tokenizer.json
curl -L -o pocket_multilingual.model https://huggingface.co/Thanushttk/pocket-tts-multilingual/resolve/main/tokenizer.model
cd ..
```

## Tokenizer study (Colab)

`notebooks/tokenizer_colab.ipynb` trains the variants in `maltok/` (normaliser, mlmorph boundaries, akshara-safe)
on `smcproject/ml-wiki-sentences` and evaluates them (`python -m maltok.evaluate`). Plan, predictions and results:
`RESEARCH_LOG.md`. Tests: `python -m pytest -q`.

## Scripts

| script | what it measures |
|---|---|
| `scripts/measure_tokens.py data/flores200_dataset data` | tokens per word, English vs Malayalam, on the same FLORES sentences |
| `scripts/inspect_pocket_tok.py data/pocket_multilingual.model data/flores200_dataset` | how the 4k multilingual Pocket TTS vocabulary is shared out between scripts; fertility per language |
| `scripts/check_ml_rules.py data/flores200_dataset data` | rule 1: noun stem\|suffix boundaries (mlmorph lemma + akshara rounding); rule 2: cuts inside an akshara |

## First results (FLORES-200 dev+devtest, 2026-11-06)

Tokens per word (Malayalam / English, same 2,009 sentences): GPT-5 o200k 3.50 / 1.22 (1.95× tokens for the same
text), GPT-4 cl100k 16.16 / 1.24 (8.94×), SMC Unigram 16k 2.13, SMC BPE 16k 2.11, Pocket multilingual 4k 4.28 / 2.59.

Grammar check on 1,447 inflected nouns (mlmorph analyses 79% of the 15,322 distinct words):

| tokenizer | clean stem\|suffix | stem itself cut | cuts inside an akshara |
|---|---|---|---|
| SMC Unigram 16k | 76.2% | 15.7% | 36.0% |
| SMC BPE 16k | 67.3% | 30.3% | 37.4% |
| Pocket multilingual 4k | 7.0% | 93.0% | 38.2% |
| GPT-5 o200k | 5.6% | 94.1% | 56.9% |

Findings:
- Many of SMC's cuts inside an akshara are at morpheme boundaries whose suffix is a vowel sign
  (`തരംതിരിക്ക|ാൻ`, `സാധിക്ക|ുന്ന`): morpheme boundary and syllable boundary conflict.
- Old ZWJ chillus (`ര്‍` instead of `ർ`) are not normalised and get shredded:
  `അ · ര്‍ · ബു · ദ · ത്തിന്` → after normalising, `അർബുദ · ത്തിന്`.

Limits: the stem boundaries are silver (constructed, not hand-checked), nouns only, mlmorph's top analysis, one corpus.

## Next

- Verbs; a larger corpus (Thottingal's 2.25M Wikipedia sentences).
- Tokenizer variants: chillu-normalised; syllable-safe; mlmorph-guided boundaries.
- Rules from public-domain grammars: Keralapāṇinīyam (1896/1917; Malayalam Wikisource), Gundert's grammar (1868)
  and dictionary (1872).

## Old letters and encoding variants (`scripts/old_letters.py data`)

The same word written in old or alternative encodings is tokenized differently, and mlmorph fails on the old forms:

| form | SMC Unigram | Pocket 4k | mlmorph |
|---|---|---|---|
| ധൎമ്മം (dot reph, old) / ധർമ്മം (chillu r) | 4 / 2 pieces | 4 / 4 | — / analysed |
| അവന്‍റെ (ZWJ) / അവന്റെ / അവൻ്റെ (three ന്റ encodings) | 3 / 1 (+▁) / 3, the last as `ൻ · ്റെ` | 2 / 2 / 2 | — / analysed / analysed |
| ൧൯൯൬ (Malayalam digits) | one piece per digit | UNK | — |
| ഩ, ഺ, ൹ (archaic letters, date mark) | kept | UNK | foreign word |

So a canonical normaliser is needed before tokenizing (and before mlmorph): dot reph → chillu r, ZWJ chillus →
atomic chillus, one ന്റ encoding, Malayalam digits → number words (mlmorph's number generator can spell them out).
