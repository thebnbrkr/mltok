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

Trains four variants of Santhosh Thottingal's recipe (HF `tokenizers` Unigram, 16k, NFC + Metaspace) on the 2.25M-sentence Malayalam Wikipedia corpus (`smcproject/ml-wiki-sentences`) and evaluates them, with the released SMC tokenizer as a reference, on FLORES-200 Malayalam and on sentences from held-out Wikipedia articles (~1% of articles, never used in training):

| variant | what is done to the text first |
|---|---|
| `base` | nothing: the SMC recipe retrained on this corpus (separates data from recipe) |
| `norm` | normaliser: 2008-era chillus → atomic chillus, three spellings of ന്റ → one |
| `morph` | `norm` + pieces never cross mlmorph's stem\\|suffix boundary |
| `akshara` | `norm` + pieces are whole aksharas (each akshara mapped to one private-use character) |

Predictions are written in `RESEARCH_LOG.md` before this runs.

**Runtime:** CPU only. A runtime with several CPUs and high RAM is best (mlmorph analysis runs in parallel; Unigram training on 2.25M sentences needs several GB). Results go to `/content/results`; the last cell downloads them.
"""),
    code("""
import os, sys, subprocess
REPO_URL = 'https://github.com/thebnbrkr/malayalam-tok.git'
REPO, DATA, OUT = '/content/malayalam-tok', '/content/data', '/content/results'
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
if not os.path.exists(f'{DATA}/flores200_dataset'):
    !cd {DATA} && curl -s -L -o flores.tar.gz https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz && tar -xzf flores.tar.gz flores200_dataset/dev/mal_Mlym.dev flores200_dataset/devtest/mal_Mlym.devtest && rm flores.tar.gz
SMC = hf_hub_download('smcproject/malayalam-unigram-tokenizer', 'tokenizer.json', local_dir=f'{DATA}/smc')
WIKI = hf_hub_download('smcproject/ml-wiki-sentences', 'ml.sentences.parquet', repo_type='dataset', local_dir=DATA)
!ls -la {DATA}
"""),
    md("""
## Stage 1: pilot on 200k sentences

Checks that everything runs and how long each variant takes. Pilot results are not used for conclusions.
"""),
    code("""
!python -m maltok.train --data {WIKI} --max-sentences 200000 --out {OUT}/pilot --skip-existing
!python -m maltok.evaluate --flores {DATA}/flores200_dataset --smc {SMC} --wiki {WIKI} --models {OUT}/pilot --out {OUT}/pilot_eval
"""),
    md("## Stage 2: full corpus (2.25M sentences), 16k vocabulary"),
    code("""
!python -m maltok.train --data {WIKI} --out {OUT}/full --skip-existing
!python -m maltok.evaluate --flores {DATA}/flores200_dataset --smc {SMC} --wiki {WIKI} --models {OUT}/full --out {OUT}/full_eval
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
