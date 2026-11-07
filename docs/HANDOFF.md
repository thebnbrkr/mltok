# mltok hand-off: a better Malayalam tokenizer, and then a Malayalam Pocket TTS

This document is for someone picking up the project to run it on a GPU machine or in Colab. It covers what we
are trying to do and why, what has been measured so far, how the code is organised, how to run it, how to
change it, what is still open, and where everything Malayalam-specific comes from.

- **Code:** https://github.com/thebnbrkr/mltok (MIT)
- **Colab notebook:** https://colab.research.google.com/github/thebnbrkr/mltok/blob/main/notebooks/tokenizer_colab.ipynb
- **Diary of every run, prediction and result:** [`RESEARCH_LOG.md`](../RESEARCH_LOG.md). Entries are appended,
  never rewritten; predictions are written there before each run.

---

## 1. Goal and rationale

**Goal.** Build a Malayalam tokenizer that is better than the best existing one (Santhosh Thottingal's SMC
Unigram tokenizer, `smcproject/malayalam-unigram-tokenizer`), and then test whether a better tokenizer makes a
better Malayalam text-to-speech model (Kyutai's Pocket TTS).

**Why the tokenizer matters.**
- Malayalam is agglutinative: case, number and tense endings, and whole words, are glued together, with sound
  changes at the joins (സന്ധി, sandhi) and compounding (സമാസം, samasam). Words are long, so tokenizers break
  them into many pieces.
- General tokenizers do this badly. On FLORES-200 (the same 2,009 sentences in English and Malayalam), GPT-4's
  tokenizer needs **16.2 tokens per Malayalam word** and 8.9× as many tokens as English for the same text; GPT-5's
  needs 3.5 per word (1.95× English).
- The multilingual Pocket TTS model (community fine-tune by Thanush, `Thanushttk/pocket-tts-multilingual`) shares
  a 4,000-piece vocabulary across 7 languages. Malayalam gets 601 pieces, mostly fragments of syllables (4.28
  tokens per word). It has the worst word error rate of its languages (23.7%), though its author attributes this
  to data, so tokenizer and data cannot be separated there.

**Why build on Thottingal's tokenizer rather than start over.** It is already the best: 2.13 tokens per word on
FLORES, and it cuts 76% of inflected nouns cleanly at stem | suffix, against 6–7% for GPT-5's and the Pocket
tokenizer. We keep his exact recipe and fix measured weaknesses, so every difference can be traced to one change.

**Research framing.** The question is whether grammar-aware tokenization (stems, suffixes, compounds, sandhi)
helps, measured fairly. Our Hungarian ASR paper (IWCLUL 2026, accepted) found that a morphology-aligned
vocabulary did *not* improve recognition accuracy, so the design allows a negative result; a positive result is
then believable. "Better" is pre-registered in `RESEARCH_LOG.md`: better on the grammar and robustness measures
without more than 5% more tokens per word.

**Credit.** Thottingal's tokenizer recipe, the mlmorph analyser and the Wikipedia sentence corpus come from him
and Swathanthra Malayalam Computing (SMC). He proposed a morphology-informed tokenizer in his blog but did not
test it. Cite him.

---

## 2. What has been measured so far

All numbers are on FLORES-200 Malayalam (dev + devtest) unless marked "Wiki" (5,000 sentences from held-out
Wikipedia articles). Details and every run are in `RESEARCH_LOG.md`.

### Weaknesses found in Thottingal's released tokenizer

| weakness | measurement |
|---|---|
| 2008-era spellings not normalised (chillus typed as consonant + virama + ZWJ before Unicode 5.1; three spellings of ന്റ) | 96 of 2,009 sentences affected; words written the 2008 way are cut the same way only 41.3% of the time (5,203 words) |
| invisible characters | a stray zero-width space keeps the same cut in only 24.9% of words; a ZWNJ inside a conjunct in 3.4% |
| typos | one missing vowel sign adds 0.78 tokens per word; 19.9% of words gain 2+ tokens |
| lone word-start token `▁` | 4.2% of its tokens (on 8.9% of words); without them 2.12 → 2.03 tokens per word |
| vocabulary spent on foreign characters | 3,981 of its 16,000 pieces (25%) are single non-Malayalam characters (1,446 Chinese) |
| verbs | only 26% of inflected verbs cut cleanly at stem \| suffix |

### Our variants so far (Unigram, 16k vocabulary)

Pilot, 200k training sentences:

| tokenizer | tok/word (FLORES / Wiki) | nouns clean | verbs clean | cuts inside an akshara |
|---|---|---|---|---|
| Thottingal, released | 2.13 / 2.36 | 75.8% | 26.2% | 36.1% |
| base | 2.13 / 2.15 | 74.5% | 26.3% | 34.0% |
| norm | 2.11 / 2.15 | 74.3% | 26.3% | 33.9% |
| morph | 2.17 / 2.18 | 86.7% | 76.8% | 29.3% |
| akshara | 2.38 / 2.39 | 64.6% | 39.2% | 0.5% |

Full corpus (2.23M sentences, **before** the character-coverage fix): every variant got *less* compact (base
2.36 tok/word), because the trainer put 7,640 single foreign characters into the vocabulary (48%). morph was
still far ahead on verbs (66–70% clean). The fix (character coverage, below) is in the next run.

**Reading so far:** morph is the best candidate (much better grammar for 1–3% more tokens); norm is a free
robustness fix; akshara (never cut inside a syllable) costs 11–13% more tokens and does worse on grammar,
because Malayalam suffixes often begin inside a syllable (തരംതിരിക്ക|ാൻ).

### First look at the പിരിച്ചെഴുതുക gold set (26 words, indicative only)

| tokenizer | joins found | clean | doubling | insertion | substitution | elision |
|---|---|---|---|---|---|---|
| Thottingal, released | 73% | 69% | 100% | 25% | 50% | 75% |
| ours, full run | 85% | 77% | 100% | 25% | 75% | 100% |

---

## 3. The design

### Text preparation (`maltok/text.py`)

- **Normaliser:** NFC; the three spellings of ന്റ (ന്‍റ, ൻ്റ, ന്റ) → ന്റ (na + virama + rra); 2008-era chillus
  (ണ്‍ ന്‍ ര്‍ ല്‍ ള്‍ ക്‍) → atomic chillus (ൺ ൻ ർ ൽ ൾ ൿ). Archaic letters are left alone (by decision: modern
  text only).
- **Akshara** (orthographic syllable): a consonant or consonant cluster joined by virama, plus its vowel sign,
  anusvara or visarga (വീട്ടിൽ = വീ · ട്ടി · ൽ). A virama followed by ZWJ or ZWNJ ends an akshara.

### Word boundaries (`maltok/morph.py`, all "silver", i.e. generated, not hand-checked)

1. **Stem | suffix** from mlmorph: for a single-root analysis (മരം<n><locative>), the stem ends where the word
   stops matching its lemma (മര|ത്തിൽ). A cut that would split a consonant cluster moves back to the akshara start.
2. **Compounds and sandhi** from mlmorph's multi-root analyses (ആന<n><adj>പുറം<n><locative>): each later root is
   located by the start of its surface form; a doubled consonant or an inserted glide goes with the second part:
   ആന | പ്പുറ | ത്തു, തരം | തിരിക്ക | ാൻ, മഴ | യില്ല.
3. **Whole word first, then a dictionary-checked fallback** (for the ~21% of words mlmorph cannot analyse): if a
   string is a word by itself, keep it whole (കുട്ടി; തീപ്പെട്ടി, although mlmorph also knows തീ + പെട്ടി).
   Otherwise the reversed sandhi formulas propose splits with the parts restored, and a split is accepted only if
   both parts are known words (mlmorph analyses them, or they occur at least `LEX_MIN = 5` times as words in the
   training corpus). Example: കാട്ടാന → കാട് + ആന (cut കാ | ട്ടാന). At most two parts.

| sandhi | joining | reversed (splitting) formula | example |
|---|---|---|---|
| ലോപം elision | C് + V → C·Vsign | C·Vsign ⇒ C് + V (also ു, chillu and anusvara forms) | കണ്ടു + ഇല്ല → കണ്ടില്ല |
| ആഗമം insertion | V + V → V·(യ\|വ)·Vsign | (യ\|വ)·Vsign ⇒ + V | മഴ + ഇല്ല → മഴയില്ല |
| ദ്വിത്വം doubling | V + C → V·C്C | C്C ⇒ + C; with elision C്C·Vsign ⇒ C് + V | ആന + പുറം → ആനപ്പുറം |
| ആദേശം substitution | ൽ + മ → ന്മ, ൺ + ന → ണ്ണ, ം + ത → ന്ത, ം + പ → മ്പ | the reverse | നൽ + മ → നന്മ |

### Tokenizer variants (`maltok/pipeline.py`)

All use Thottingal's recipe: Hugging Face `tokenizers` Unigram, 16k vocabulary, Strip + NFC normaliser,
Metaspace pre-tokenizer (prepend always), special tokens `<s> <pad> </s> <unk> <mask>`. Forced boundaries are
inserted as an invisible separator (U+2063) that the pre-tokenizer splits on and removes.

| variant | text preparation |
|---|---|
| `base` | none (his recipe retrained on our data) |
| `norm` | normaliser |
| `morph` | normaliser + stem \| suffix boundaries |
| `compound` | `morph` + compound/sandhi joins + the whole-word-first fallback (saves its corpus lexicon in `lexicon.txt`) |
| `akshara` | normaliser + each frequent akshara mapped to one private-use character, so pieces are whole aksharas |

**Character coverage** (`--char-coverage 0.9999`): keep the characters making up 99.99% of character
occurrences (Malayalam letters, ASCII, ZWJ and ZWNJ always kept) and drop the rest from training. Without it,
the trainer gives every character it sees a vocabulary slot.

### Data

- **Training:** `smcproject/ml-wiki-sentences` (Thottingal; 2.25M Malayalam Wikipedia sentences, CC-BY-SA).
  Articles whose URL hashes to 0 mod 100 (~1%) are held out and never trained on (2,230,143 training sentences).
- **Evaluation:** FLORES-200 Malayalam dev + devtest (2,009 translated news sentences); 5,000 sentences from the
  held-out Wikipedia articles; the gold sets below.

### Evaluation (`maltok/evaluate.py`, `maltok/gold.py`, `maltok/devadath.py`)

| measure | meaning |
|---|---|
| tok/word | tokens per whitespace word (lower = shorter sequences) |
| lone ▁ | share of tokens that are only the word-start marker |
| unk, Rényi efficiency | unknown tokens; how evenly the vocabulary is used (Zouhar et al. 2023) |
| nouns / verbs clean | of the words a tokenizer splits, the share cut at stem \| suffix without chopping the stem (partly circular for morph/compound, which are trained on the same boundaries) |
| compounds clean | the same for mlmorph compounds, at the first join |
| cuts inside an akshara | share of cuts that land inside a syllable |
| robustness | same cut under 2008 spellings, a stray zero-width space, a ZWNJ in a conjunct; extra tokens after a one-vowel-sign typo |
| gold: joins found / clean | on `gold/pirichezhuthuka.tsv` (draft, 26 words, sandhi type per row; `whole_ok = yes` rows may stay whole) |
| Devadath: point / word recall | on Devadath et al.'s 13,636 gold words and 999 validation words (external sandhi only; downloaded at run time, GPL-3.0, not redistributed); both glide conventions count |

---

## 4. Repository layout

```
maltok/text.py        normaliser, akshara rules, 2008-spelling generator for tests
maltok/morph.py       mlmorph boundaries, compounds/sandhi, whole-word-first fallback
maltok/pipeline.py    tokenizer variants, character coverage, save/load
maltok/train.py       CLI: train variants on the Wikipedia corpus (held-out split by article)
maltok/evaluate.py    CLI: all measures, tables in eval.md, everything in eval.json
maltok/gold.py        പിരിച്ചെഴുതുക gold set scoring
maltok/devadath.py    Devadath et al. gold data: download, load, recall scoring
gold/pirichezhuthuka.tsv   the gold set (draft; edit freely, see §6)
notebooks/make_notebook.py generates notebooks/tokenizer_colab.ipynb (edit the generator, not the .ipynb)
scripts/              the early one-off measurements (fertility vs English, Pocket vocabulary, old letters)
tests/test_maltok.py  23 toy-sized tests (seconds): run before every change
RESEARCH_LOG.md       the diary
```

---

## 5. How to run it

### In Colab

Open the notebook link above, choose a CPU runtime (high-RAM if available; a GPU is not used for tokenizers),
Run all. Stages 1 and 2 are off by default (their results are logged); Stage 3 trains `norm, compound, morph,
base, akshara` with character coverage and evaluates everything. Finished tokenizers are skipped on re-run. The
last cell downloads `results.zip`. Measured on a 12-CPU Colab runtime, full corpus: base ~12 min, norm ~13,
morph ~20, akshara ~9 (compound not yet measured). A 2-CPU T4 runtime is several times slower; set
`MAX_SENTENCES = 1_000_000` in the Stage 3 cell to roughly halve it.

### On your own machine

```bash
git clone https://github.com/thebnbrkr/mltok && cd mltok
pip install -r requirements.txt
python -m pytest -q                                   # 23 tests, seconds

# data (FLORES archive members start with "./"; extract the two Malayalam files)
mkdir -p data && python - <<'PY'
import tarfile, urllib.request, os
urllib.request.urlretrieve("https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz", "data/flores.tar.gz")
with tarfile.open("data/flores.tar.gz") as t:
    for m in t.getmembers():
        if m.name.endswith(("/dev/mal_Mlym.dev", "/devtest/mal_Mlym.devtest")):
            m.name = m.name.lstrip("./"); t.extract(m, "data")
os.remove("data/flores.tar.gz")
from huggingface_hub import hf_hub_download
hf_hub_download("smcproject/malayalam-unigram-tokenizer", "tokenizer.json", local_dir="data/smc")
hf_hub_download("smcproject/ml-wiki-sentences", "ml.sentences.parquet", repo_type="dataset", local_dir="data")
PY

# train (add --max-sentences N for a quicker run; --skip-existing to resume)
python -m maltok.train --data data/ml.sentences.parquet --char-coverage 0.9999 \
    --variants norm compound morph base akshara --out results/full_cov --skip-existing

# evaluate (Devadath's data is downloaded into data/devadath)
python -m maltok.evaluate --flores data/flores200_dataset --wiki data/ml.sentences.parquet \
    --smc data/smc/tokenizer.json --devadath data/devadath --models results/full_cov --out results/full_cov_eval
```

Tokenizer training is CPU work and uses all cores (mlmorph analysis runs in parallel). It needs several GB of
RAM for the full corpus. The GPU matters only for the TTS stage (§8).

---

## 6. How to change things

| to change | where |
|---|---|
| vocabulary size | `--vocab` (default 16000) |
| character coverage | `--char-coverage` (omit for Thottingal's every-character behaviour) |
| training size | `--max-sentences` (random sample, seed 0) |
| normaliser rules | `maltok/text.py: normalize()`; add a case to `test_normalize_2008_spellings` |
| a new variant | `maltok/pipeline.py`: add the name to `VARIANTS`, its preparation in `Pipeline.prepare`, any pre-tokenizer change in `new_tokenizer`, any training-time setup in `train()`; add a test |
| a different algorithm (e.g. BPE) | `new_tokenizer()` (model) and `train()` (trainer); keep the same normaliser and pre-tokenizer so only one thing changes |
| sandhi formulas | `maltok/morph.py: candidates()`; each candidate is (surface cut, rule name, restored left part, restored right part) |
| how strict the "known word" check is | `morph.LEX_MIN` (corpus count) and `morph.known()` |
| where a glide or doubled consonant goes | `morph.compound()` (the `b -= 1` glide rule) and `gold.joins()`; decide with a native speaker first |
| gold set | `gold/pirichezhuthuka.tsv`: tab-separated `word, parts (joined by " + "), sandhi, whole_ok (yes/no), note`; lines starting with `#` are comments |
| measures and table columns | `maltok/evaluate.py`: `evaluate()` computes, `MAIN` and `ROBUST` choose the columns |
| the notebook | edit `notebooks/make_notebook.py`, then `python notebooks/make_notebook.py` |

Please log every run in `RESEARCH_LOG.md` (what was run, the commit, the numbers, and the predictions you wrote
before running), and keep the tests passing.

---

## 7. Open decisions and known limits

- **Native-speaker gold set (most important).** Check the 26 rows of `gold/pirichezhuthuka.tsv` and the
  `whole_ok` marks; add more പിരിച്ചെഴുതുക examples (insertion and substitution especially, ideally from school
  textbooks); build ~300 words of stem | suffix boundaries. The silver boundaries are our construction.
- **Glide convention.** Ours: inserted യ/വ goes right (മഴ | യില്ല). Devadath's data: left (തുകയ | ാണ്). Both are
  scored as correct for now.
- **Lexicalised compounds** stay whole by rule (തീപ്പെട്ടി, തിരുവോണം). Devadath's data splits more aggressively
  (അതു | കൊണ്ട | ാണ്). Pick one convention for the paper.
- **Circularity:** morph/compound are scored on the same mlmorph boundaries they were trained on. The gold sets
  are the fair test.
- **Fallback limits:** two parts at most; misses some elisions (രാപ്പകൽ, where രാവ് loses its വ്).
- **Not fixed yet:** stray zero-width spaces (predicted to hurt all variants; remove in the normaliser if
  confirmed); the lone ▁ token.
- **Pocket TTS** loads a SentencePiece `.model`; our tokenizers are Hugging Face `tokenizer.json`. Its text
  conditioner (`pocket_tts/conditioners/text.py`, `LUTConditioner`) needs a small adapter, or the chosen variant
  is re-trained with the `sentencepiece` library on the same prepared text.
- **Licences:** our code MIT; mlmorph and the SMC tokenizer MIT; ml-wiki-sentences CC-BY-SA; Devadath's data
  GPL-3.0 (downloaded at run time, never committed); InstaDeep's Plants Genomic Benchmark is irrelevant here.

---

## 8. Next steps

1. **Stage 3 run** (Colab or your machine): all variants with character coverage, full evaluation. Compare with
   the predictions in `RESEARCH_LOG.md`.
2. **Native-speaker checks** (§7).
3. **Fixes and baselines:** remove stray ZWSP if confirmed; fix the lone ▁; add BPE, character-level, syllable
   (Mlphon's syllabifier) and Morfessor baselines.
4. **Malayalam Pocket TTS** with the 2–3 best tokenizers, same data and steps, then the Pocket-hu evaluation
   (two ASR judges, paired bootstrap over sentences, error categories, a Malayalam minimal-pair test: ല/ള/ഴ,
   ന/ണ, consonant length as in കല vs കല്ല, vowel length).
   - Kyutai's training code: `training/` in kyutai-labs/pocket-tts (≥100 h of speech, 1,000+ recommended; word
     alignments from a CTC wav2vec2 model for the language; `finetune_language.yaml` starts from the English
     teacher with a fresh text embedding; `depth_distill.yaml` distils the 24-layer teacher into a 6-layer
     student). In Hungarian (Pocket-hu) the teacher reached its final WER after ~5k steps.
   - Open Malayalam speech: IMaSC (~49.6 h, 8 speakers, studio), IndicTTS Malayalam (~17.9 h, 2 speakers),
     IndicVoices-R Malayalam (~82.6 h, 462 speakers, mostly spontaneous): ~150 h in all. AI4Bharat's Shrutilipi
     (All India Radio) may add more (not checked).

**Possible venues:** LInDA workshop at EACL 2027 (15 December 2026; a linguistic resource paired with its effect
on a task: fits mlmorph → tokenizer → TTS), Interspeech 2027 (~9 February 2027, with TTS results), ACL 2027 via
ARR (~4 January 2027, if the TTS results are ready), DravidianLangTech 2027 (call not out yet). Confirm dates on
the official sites.

---

## 9. References

### Malayalam language technology (SMC, Thottingal)

- Thottingal, "The Broken Token: Tokenization for Malayalam Language Models" (Feb 2026):
  https://thottingal.in/blog/2026/02/27/malayalam-tokenizer-llm/ · code: https://github.com/smc/malayalam-tokenizer
- Thottingal, "What Do MTEB Scores Mean?" (Oct 2026; Malayalam benchmark scores are 84% one FLORES task):
  https://thottingal.in/blog/2026/10/09/mteb-scores/
- Thottingal, "Why your slug generator is dropping Malayalam vowels" (Sep 2026; code that drops combining marks):
  https://thottingal.in/blog/2026/09/18/slug-generators-and-vowels-signs/
- Thottingal, "Preparing sentence dataset from a wikipedia" (Mar 2026): https://thottingal.in/blog/2026/03/14/wikisentences/
- mlmorph (FST morphological analyser, MIT): https://github.com/smc/mlmorph · demo https://morph.smc.org.in ·
  Thottingal 2019, "Finite State Transducer based Morphology analysis for Malayalam Language" (W19-6801).
  Its reference list (https://morph.smc.org.in/reference.html) names the grammars below.
- Tokenizers: https://huggingface.co/smcproject/malayalam-unigram-tokenizer ·
  https://huggingface.co/smcproject/malayalam-bpe-tokenizer · dataset https://huggingface.co/datasets/smcproject/ml-wiki-sentences
- Mlphon (Kavya Manohar; Malayalam phonetic analyser, syllabification, G2P): https://pypi.org/project/mlphon/
- Manohar, Jayan & Rajan 2023, subword tokens for Malayalam ASR (syllable-BPE best, 10.6% WER), EURASIP:
  https://link.springer.com/article/10.1186/s13636-023-00313-7

### Grammars

- A. R. Raja Raja Varma, *Keralapāṇinīyam* (1896, revised 1917; sandhi and samasam chapters):
  https://ml.wikisource.org/wiki/കേരളപാണിനീയം
- Ravi Sankar S. Nair, *A Grammar of Malayalam*, Language in India 12:11 (2012), free PDF at languageinindia.com
- R. E. Asher & T. C. Kumari, *Malayalam* (Routledge Descriptive Grammars, 1997): the standard reference
- H. Gundert, *Malayāḷabhāṣā Vyākaraṇam* (1868, archive.org `dli.mal.CiXIV68a`) and *Malayalam–English
  Dictionary* (1872, searchable at https://dsal.uchicago.edu/dictionaries/gundert/)

### Sandhi splitting

- Nair & Peter 2011, "Development of a Rule Based Learning System for Splitting Compound Words in Malayalam
  Language", IEEE RAICS, pp. 751–755 (rules + trie; ~90% on 4,000 compounds; no public code found)
- Devadath, Kurisinkel, Sharma & Varma 2014, "A Sandhi Splitter for Malayalam", ICON:
  https://aclanthology.org/W14-5125/ (91.1% on 260 sandhi words) · data and code:
  https://github.com/Devadath/MLP_Word_Segementation (GPL-3.0)
- Kuncham et al. 2015, "Statistical Sandhi Splitter for Agglutinative Languages" (CRF; 90.5% Malayalam)
- Girija & Anuradha 2017, "Application of Finite State Methods in Malayalam Text Analysis", IJCA 168(12)
  (FST handling of the four sandhi types)
- Sandhi-1.0-8M (2026, Apache-2.0; trained on mlmorph/Mlphon labels): https://huggingface.co/nithinmanoj10/Sandhi-1.0-8M
- Tamil analyser for a later Dravidian extension: ThamizhiMorph, https://github.com/sarves/thamizhi-morph

### Tokenizer evaluation

- Arnett & Bergen 2025, MorphScore; "Evaluating Morphological Alignment of Tokenizers in 70 Languages":
  https://arxiv.org/pdf/2507.06378
- Zouhar et al. 2023, Rényi efficiency
- "The Tokenizer Tax" for Indian languages (Malayalam 13× English on cl100k): https://arxiv.org/abs/2607.24276
- Our Hungarian work: the IWCLUL 2026 ASR paper (tokenizer study: Unigram/BPE/Morfessor vocabularies; boundary F1
  and MorphScore) and the Pocket-hu TTS paper (evaluation protocol for §8)

### Data and models

- FLORES-200: https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz
- Sangraha (AI4Bharat; Malayalam ~16.4B tokens, ~2.7B verified): https://huggingface.co/datasets/ai4bharat/sangraha
- Pocket TTS: https://github.com/kyutai-labs/pocket-tts (training README in `training/`) · multilingual
  community model: https://huggingface.co/Thanushttk/pocket-tts-multilingual
- Speech: IMaSC (LREC 2026) https://aclanthology.org/2026.lrec-1.465/ · IndicTTS Malayalam
  https://huggingface.co/datasets/SPRINGLab/IndicTTS_Malayalam · IndicVoices-R https://arxiv.org/pdf/2409.05356
