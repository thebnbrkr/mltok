"""Write notebooks/tokenizer_colab.ipynb.

    python notebooks/make_notebook.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.strip("\n").splitlines(keepends=True)}


CELLS = [
    md("""
# Malayalam tokenizer: can we improve on the SMC Unigram tokenizer?

Trains five variants of Santhosh Thottingal's recipe (HF `tokenizers` Unigram, 16k, NFC + Metaspace) on the 2.25M-sentence Malayalam Wikipedia corpus (`smcproject/ml-wiki-sentences`) and evaluates them, with the released SMC tokenizer as a reference, on FLORES-200 Malayalam and on sentences from held-out Wikipedia articles (~1% of articles, never used in training):

| variant | what is done to the text first |
|---|---|
| `base` | nothing: the SMC recipe retrained on this corpus (separates data from recipe) |
| `norm` | normaliser: 2008-era chillus → atomic chillus, three spellings of ന്റ → one |
| `morph` | `norm` + pieces never cross mlmorph's stem\\|suffix boundary |
| `compound` | `morph` + pieces never cross compound and sandhi joins (സമാസം, സന്ധി: ആന\\|പ്പുറ\\|ത്തു); words mlmorph does not know are split only if both parts are real words (കാട്ടാന → കാട് + ആന) |
| `akshara` | `norm` + pieces are whole aksharas (each akshara mapped to one private-use character) |

Predictions are written in `RESEARCH_LOG.md` before this runs.

**Runtime:** CPU only; a GPU is not used. A T4 runtime works but has 2 CPUs, so the full run takes a few hours (several CPUs and high RAM are faster). Stages 1 and 2 are off by default: their results are logged in `RESEARCH_LOG.md`. Stage 3 is the main run. Finished tokenizers are skipped on re-run, so after a disconnect just run the cells again (upload the last `results.zip` to `/content` and unzip it first if the runtime was deleted). Results go to `/content/results`; the last cell downloads them.
"""),
    code("""
import os, sys, subprocess
REPO_URL = 'https://github.com/thebnbrkr/mltok.git'
REPO, DATA, OUT = '/content/mltok', '/content/data', '/content/results'
os.makedirs(DATA, exist_ok=True); os.makedirs(OUT, exist_ok=True)
if os.path.exists(REPO):
    !git -C {REPO} pull -q
else:
    !git clone -q {REPO_URL} {REPO}
%cd {REPO}
!git log --oneline -1
!pip install -q mlmorph tokenizers pyarrow huggingface_hub pytest
for _m in [k for k in sys.modules if k == 'maltok' or k.startswith('maltok.')]:
    del sys.modules[_m]
print('CPUs:', os.cpu_count())
"""),
    md("Unit tests (toy-sized; seconds). Stop if anything fails."),
    code("!python -m pytest -q"),
    md("Data: FLORES-200 (evaluation only), the released SMC tokenizer, and the Wikipedia sentences (training)."),
    code("""
from huggingface_hub import hf_hub_download
import tarfile, urllib.request
FLORES = f'{DATA}/flores200_dataset'
if not os.path.exists(f'{FLORES}/dev/mal_Mlym.dev'):
    tgz = f'{DATA}/flores.tar.gz'
    urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz', tgz)
    with tarfile.open(tgz) as t:   # member names start with './'; match on the end so any tar layout works
        for m in t.getmembers():
            if m.name.endswith(('/dev/mal_Mlym.dev', '/devtest/mal_Mlym.devtest')):
                m.name = m.name.lstrip('./')
                t.extract(m, DATA)
    os.remove(tgz)
for f in ('dev/mal_Mlym.dev', 'devtest/mal_Mlym.devtest'):
    n = sum(1 for _ in open(f'{FLORES}/{f}', encoding='utf-8'))
    assert n > 900, f'{f}: only {n} lines'
    print(f, n, 'lines')
SMC = hf_hub_download('smcproject/malayalam-unigram-tokenizer', 'tokenizer.json', local_dir=f'{DATA}/smc')
WIKI = hf_hub_download('smcproject/ml-wiki-sentences', 'ml.sentences.parquet', repo_type='dataset', local_dir=DATA)
!ls -la {DATA}
"""),
    md("""
## Stage 1: pilot on 200k sentences (off by default; results logged)

Checks that everything runs and how long each variant takes. Pilot results are not used for conclusions.
"""),
    code("""
RUN_PILOT = False
if RUN_PILOT:
  !python -m maltok.train --data {WIKI} --max-sentences 200000 --out {OUT}/pilot --skip-existing
  !python -m maltok.evaluate --flores {DATA}/flores200_dataset --smc {SMC} --wiki {WIKI} --models {OUT}/pilot --out {OUT}/pilot_eval
"""),
    md("## Stage 2: full corpus without character coverage (off by default; results logged)"),
    code("""
RUN_OLD_FULL = False
if RUN_OLD_FULL:
  !python -m maltok.train --data {WIKI} --out {OUT}/full --skip-existing
  !python -m maltok.evaluate --flores {DATA}/flores200_dataset --smc {SMC} --wiki {WIKI} --models {OUT}/full --out {OUT}/full_eval
"""),
    md("""
## Stage 3: full corpus with 99.99% character coverage

The full-corpus run above spends about half of its 16k vocabulary on single foreign characters (Chinese, Korean, Arabic, ...), because the trainer adds every character it sees. Here the rarest characters (the last 0.01% of character occurrences) are dropped from training, as SentencePiece's `character_coverage` does. This stage also trains the new `compound` variant and scores every tokenizer on the പിരിച്ചെഴുതുക gold set (`gold/pirichezhuthuka.tsv`, a draft) and on Devadath et al.'s sandhi gold data (downloaded at run time; GPL-3.0, not redistributed), with the harder robustness tests. Variants are trained in order of importance (norm, compound, morph, base, akshara), so the key ones finish first. On a T4, set `MAX_SENTENCES = 1_000_000` to roughly halve the time.
"""),
    code("""
MAX_SENTENCES = None            # None = all 2.23M training sentences; e.g. 1_000_000 on a slow runtime
limit = f'--max-sentences {MAX_SENTENCES}' if MAX_SENTENCES else ''
!python -m maltok.train --data {WIKI} --char-coverage 0.9999 --variants norm compound morph base akshara {limit} --out {OUT}/full_cov --skip-existing
models = ' '.join(d for d in (f'{OUT}/full', f'{OUT}/full_cov') if os.path.isdir(d))
!python -m maltok.evaluate --flores {DATA}/flores200_dataset --wiki {WIKI} --smc {SMC} --devadath {DATA}/devadath --models {models} --out {OUT}/full_cov_eval
"""),
    md("""
## How to read it (from `RESEARCH_LOG.md`)

- **Better** means better on the morphology and robustness columns **without** a meaningfully worse tokens-per-word (within 5%).
- `base` vs `smc-released`: effect of the training corpus alone. `norm` vs `base`: the normaliser. `morph` / `akshara` vs `norm`: the boundary rule.
- Rule 1 for `morph` is partly circular (it was trained on the same mlmorph boundaries); a native-speaker gold set is the fair test.
"""),
    md("Download the results before the runtime is deleted."),
    code("""
!cd /content && zip -qr results.zip results
from google.colab import files
files.download('/content/results.zip')
"""),
]


if __name__ == "__main__":
    nb = {"cells": CELLS, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 0}
    (HERE / "tokenizer_colab.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print("wrote tokenizer_colab.ipynb", len(CELLS), "cells")
