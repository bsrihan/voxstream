import pytest

from spc.textnorm import normalize_text, word_error_rate


def test_normalize_lowercase_and_punctuation():
    assert normalize_text("The stale smell, of OLD beer!") == \
        "the stale smell of old beer"


def test_normalize_keeps_word_internal_apostrophes():
    assert normalize_text("we've done it") == "we've done it"
    assert normalize_text("'quoted'") == "quoted"


def test_normalize_unicode_apostrophe():
    assert normalize_text("we\u2019ve") == "we've"


def test_wer_identical_is_zero():
    r = word_error_rate("a b c", "a b c")
    assert r["wer"] == 0.0
    assert r["substitutions"] == r["insertions"] == r["deletions"] == 0


def test_wer_case_and_punctuation_ignored():
    assert word_error_rate("Hello, world!", "hello world")["wer"] == 0.0


def test_wer_counts_operations():
    # ref: "the cat sat" / hyp: "a cat sat down" -> 1 sub + 1 ins
    r = word_error_rate("the cat sat", "a cat sat down")
    assert r["substitutions"] == 1
    assert r["insertions"] == 1
    assert r["deletions"] == 0
    assert r["wer"] == pytest.approx(2 / 3)


def test_wer_all_deleted():
    r = word_error_rate("one two three", "")
    assert r["deletions"] == 3
    assert r["wer"] == 1.0


def test_wer_empty_reference_raises():
    with pytest.raises(ValueError):
        word_error_rate("", "something")


def test_wer_matches_jiwer():
    """Cross-check our implementation against jiwer on a nontrivial pair."""
    jiwer = pytest.importorskip("jiwer")
    ref = "the stale smell of old beer lingers it takes heat to bring out the odor"
    hyp = "a stale smell of all beer lingers takes heat to bring out the odors"
    ours = word_error_rate(ref, hyp, normalize=False)["wer"]
    theirs = jiwer.wer(ref, hyp)
    assert ours == pytest.approx(theirs)
