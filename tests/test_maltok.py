"""Unit tests: normaliser, aksharas, mlmorph boundaries, tokenizer variants (toy-sized), evaluation."""
import pytest

from maltok import text
from maltok.text import aksharas, normalize, old_encoding

mlmorph = pytest.importorskip("mlmorph")
from maltok import morph  # noqa: E402
from maltok.evaluate import evaluate, word_cuts  # noqa: E402
from maltok.pipeline import VARIANTS, Pipeline, train  # noqa: E402

SENTENCES = [
    "മരത്തിൽ ഒരു പക്ഷി ഇരിക്കുന്നു.", "കുട്ടികൾക്ക് പുസ്തകങ്ങൾ വേണം.", "രാജ്യങ്ങളുടെ നേതാക്കൾ യോഗം ചേർന്നു.",
    "അവന്റെ വീട്ടിൽ ഇന്ന് ആഘോഷമാണ്.", "സർക്കാരിന്റെ പുതിയ പദ്ധതി പ്രഖ്യാപിച്ചു.", "നഗരത്തിൽ മഴ പെയ്തു.",
    "ആശുപത്രിയിൽ രോഗികൾ കാത്തിരുന്നു.", "മാധ്യമങ്ങൾ വാർത്ത നൽകി.", "അവർ കേരളത്തിന്റെ ചരിത്രം പഠിച്ചു.",
] * 30


def test_normalize_2008_spellings():
    assert normalize("അവരില്‍") == "അവരിൽ"                      # ZWJ chillu -> atomic
    assert normalize("അര്‍ബുദം") == "അർബുദം"
    for v in ("അവന്‍റെ", "അവൻ്റെ", "അവന്റെ"):                  # three spellings of ന്റ
        assert normalize(v) == "അവന്റെ"
    for w in ("അവന്റെ", "അവരിൽ", "കർമ്മം"):
        assert normalize(old_encoding(w)) == w and normalize(normalize(w)) == normalize(w)


def test_aksharas():
    assert aksharas("വീട്ടിൽ") == ["വീ", "ട്ടി", "ൽ"]
    assert aksharas("പ്രഖ്യാപിച്ചു") == ["പ്ര", "ഖ്യാ", "പി", "ച്ചു"]
    assert aksharas("അര്‍ബുദം") == ["അ", "ര്‍", "ബു", "ദം"]   # 2008 chillu ends an akshara
    assert text.akshara_starts("മരം") == {0, 1, 3}


def test_mlmorph_boundaries():
    assert morph.boundary("മരത്തിൽ")[:4] == ("noun", 2, 2, 2)           # മര|ത്തിൽ
    b = morph.boundary("കുട്ടികൾക്ക്")
    assert b[0] == "noun" and "കുട്ടികൾക്ക്"[:b[3]] == "കുട്ടി"
    assert morph.segment_text("ഇത് മരത്തിൽ.") == "ഇത് മര" + morph.MARK + "ത്തിൽ."
    assert morph.boundary("zzz") is None


@pytest.fixture(scope="module")
def pipes(tmp_path_factory):
    out = {}
    for v in VARIANTS:
        p = train(v, SENTENCES, vocab_size=300)
        d = tmp_path_factory.mktemp(v)
        p.save(d)
        out[v] = Pipeline.load(d)
    return out


@pytest.mark.parametrize("v", VARIANTS)
def test_pieces_spell_the_word(pipes, v):
    for w in ("മരത്തിൽ", "കുട്ടികൾക്ക്", "വീട്ടിൽ", "അവന്റെ"):
        cuts, ok = word_cuts(pipes[v], w)
        assert ok, (v, w, pipes[v].pieces(w))


def test_morph_variant_respects_boundaries(pipes):
    for w in ("മരത്തിൽ", "കുട്ടികൾക്ക്", "നഗരത്തിൽ", "രാജ്യങ്ങളുടെ"):
        b = morph.boundary(w)
        if b:
            assert b[3] in word_cuts(pipes["morph"], w)[0], (w, pipes["morph"].pieces(w))


def test_akshara_variant_never_cuts_a_mapped_akshara(pipes):
    p = pipes["akshara"]
    for s in SENTENCES[:9]:
        for w in text.MALAYALAM_RUN.findall(normalize(s)):
            cuts, ok = word_cuts(p, w)
            starts = text.akshara_starts(w)
            inside = [c for c in cuts if c not in starts]
            assert ok and not inside, (w, p.pieces(w))


def test_normalising_variants_are_robust_to_2008_spelling(pipes):
    w = "അവരിൽ"
    for v in ("norm", "morph", "akshara"):
        assert pipes[v].pieces(old_encoding(w)) == pipes[v].pieces(w)


def test_evaluate_runs(pipes):
    words = sorted({w for s in SENTENCES[:9] for w in text.MALAYALAM_RUN.findall(normalize(s)) if len(w) > 1})
    r = evaluate(pipes["norm"], SENTENCES[:9], words)
    assert r["mismatch_words"] == 0 and r["noun_n"] > 0 and 0 <= r["noun_clean"] <= 1 and r["tok_per_word"] > 0


def test_heldout_split_is_by_article_and_disjoint(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq
    from maltok.train import is_heldout, read_sentences
    urls = [f"https://ml.wikipedia.org/wiki/A{i}" for i in range(400)]
    rows = {"sentence": [f"വാക്യം {i} {j}" for i in range(400) for j in range(3)],
            "url": [u for u in urls for _ in range(3)]}
    path = tmp_path / "s.parquet"
    pq.write_table(pa.table(rows), path)
    train_lines, held = read_sentences(path), read_sentences(path, heldout=True)
    assert held and not set(train_lines) & set(held)
    assert len(train_lines) + len(held) == 1200
    assert sum(is_heldout(u) for u in urls) == len(held) // 3          # whole articles move together
