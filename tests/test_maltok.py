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


def test_perturbations_are_deterministic_and_minimal():
    from maltok.evaluate import ZWSP, with_typo, with_zwnj, with_zwsp
    w = "കുട്ടികൾക്ക്"
    assert with_zwsp(w) == with_zwsp(w) and with_zwsp(w).replace(ZWSP, "") == w
    assert with_zwnj(w).replace("‌", "") == w and "്‌" in with_zwnj(w)
    t = with_typo(w)
    assert len(t) == len(w) - 1 and t != w
    assert with_zwnj("മരം") is None and with_typo("ക") is None


def test_evaluate_reports_robustness(pipes):
    words = sorted({w for s in SENTENCES[:9] for w in text.MALAYALAM_RUN.findall(normalize(s)) if len(w) > 1})
    r = evaluate(pipes["norm"], SENTENCES[:9], words)
    for k in ("robust_zwsp", "robust_zwnj", "typo_extra", "typo_frag"):
        assert k in r


def test_char_coverage_keeps_rare_characters_out_of_the_vocabulary():
    from maltok.pipeline import covered_chars
    lines = SENTENCES + ["漢字 中文 한국어"]                       # rare foreign characters, once each
    keep = covered_chars(lines, 0.999)
    assert "漢" not in keep and "മ" in keep and "‍" in keep
    p = train("norm", lines, vocab_size=300, coverage=0.999)
    vocab = p.tok.get_vocab()
    assert not any("漢" in t or "한" in t for t in vocab)
    assert word_cuts(p, "മരത്തിൽ")[1]                           # Malayalam still spelled correctly


def test_compound_and_sandhi_joins():
    seg = lambda w: morph.segment_word(w, compounds=True).replace(morph.MARK, "|")
    assert seg("ആനപ്പുറത്തു") == "ആന|പ്പുറ|ത്തു"          # doubling: the doubled consonant goes right
    assert seg("തരംതിരിക്കാൻ") == "തരം|തിരിക്ക|ാൻ"
    assert seg("മഴയില്ല") == "മഴ|യില്ല"                    # inserted glide goes right
    assert seg("അതല്ല") == "അത|ല്ല"
    assert seg("മരത്തിൽ") == "മര|ത്തിൽ"                    # single stem falls back to stem|suffix


def test_gold_set_loads_and_places_every_join():
    from maltok import gold
    rows = gold.load()
    assert len(rows) >= 20 and all(len(r["parts"]) >= 2 for r in rows)
    assert all(gold.joins(r["word"], r["parts"]) for r in rows), [r["word"] for r in rows if not gold.joins(r["word"], r["parts"])]
    assert gold.joins("ആനപ്പുറം", ["ആന", "പുറം"]) == [(4, 2)]
    assert gold.joins("കണ്ണീർ", ["കൺ", "നീർ"])[0][1] == 1   # ആദേശം: falls back to the shared prefix (ക|ണ്ണീർ)


def test_compound_variant_trains_and_respects_joins(tmp_path):
    lines = SENTENCES + ["ആനപ്പുറത്തു കയറി.", "അവർ തരംതിരിക്കാൻ തുടങ്ങി.", "ഇന്ന് മഴയില്ല."] * 20
    p = train("compound", lines, vocab_size=300)
    cuts, ok = word_cuts(p, "ആനപ്പുറത്തു")
    assert ok and 2 in cuts                                   # ആന | പ്പുറ...
    from maltok import gold
    r = gold.evaluate(p, gold.load())
    assert r["all"]["n"] >= 20 and 0 <= r["all"]["found"] <= 1


def test_fallback_follows_the_whole_word_first_rule():
    assert morph.fallback("കുട്ടി") is None                     # a word by itself: never split
    assert morph.fallback("കാട്ടാന")[2:] == ("കാട്", "ആന")      # unknown to mlmorph; both parts are words
    assert morph.fallback("നന്മ") is None                       # ‑മ is a suffix, not a word: keep whole
    assert morph.compound("തീപ്പെട്ടി") is None                 # lexicalised: mlmorph prefers the whole word


def test_gold_whole_ok_rows_accept_keeping_the_word_whole():
    from maltok import gold
    rows = {r["word"]: r for r in gold.load()}
    assert rows["തീപ്പെട്ടി"]["whole_ok"] == "yes" and rows["ആനപ്പുറം"]["whole_ok"] == "no"


def test_devadath_format_and_scoring(tmp_path):
    from maltok import devadath
    f = tmp_path / "d"
    f.write_text("അ ത ു ക ൊ ണ ് ട ാ ണ ്\tNSP NSP SP NSP NSP NSP NSP SP NSP NSP NSP\n"
                 "ത ു ക യ ാ ണ ്\tNSP NSP NSP SP NSP NSP NSP\n", encoding="utf-8")
    items = devadath.load(f)
    assert items[0] == ("അതുകൊണ്ടാണ്", [3, 8]) and items[1] == ("തുകയാണ്", [4])
    assert devadath.hit("തുകയാണ്", 4, {3})                     # glide on the right also counts
    r = devadath.score(lambda w: {3, 8} if w.startswith("അ") else set(), items)
    assert r["point_recall"] == 2 / 3 and r["word_recall"] == 0.5


def test_evaluate_command_end_to_end(tmp_path, pipes):
    from maltok import devadath
    from maltok.evaluate import main
    fl = tmp_path / "flores"
    for split in ("dev", "devtest"):
        (fl / split).mkdir(parents=True)
        (fl / split / f"mal_Mlym.{split}").write_text("\n".join(SENTENCES[:9]), encoding="utf-8")
    models = tmp_path / "models"
    for v in ("norm", "compound"):
        pipes[v].save(models / v) if v in pipes else train(v, SENTENCES, vocab_size=300).save(models / v)
    dv = tmp_path / "dv"
    dv.mkdir()
    for f in devadath.FILES.values():                       # pre-filled, so nothing is downloaded
        (dv / f).write_text("ത ു ക യ ാ ണ ്\tNSP NSP NSP SP NSP NSP NSP\n", encoding="utf-8")
    out = tmp_path / "out"
    main(["--flores", str(fl), "--models", str(models), "--devadath", str(dv), "--out", str(out)])
    md = (out / "eval.md").read_text(encoding="utf-8")
    assert "Devadath" in md and "പിരിച്ചെഴുതുക" in md and "compounds clean" in md and "stray ZWSP" in md
