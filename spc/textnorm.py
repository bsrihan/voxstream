"""Text normalization and word error rate.

WER is computed on normalized text (lowercase, punctuation stripped, numbers
and whitespace canonicalized) so that formatting choices of the ASR/LLM stack
(casing, commas) don't count as errors — standard practice for ASR evaluation.
"""
import re
import unicodedata

import numpy as np

_APOSTROPHES = {"\u2019": "'", "\u2018": "'", "\u02bc": "'"}


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    for src, dst in _APOSTROPHES.items():
        text = text.replace(src, dst)
    text = text.lower()
    # keep letters, digits, apostrophes-in-words; everything else -> space
    text = re.sub(r"[^a-z0-9']+", " ", text)
    text = re.sub(r"(?<![a-z0-9])'|'(?![a-z0-9])", " ", text)  # bare quotes
    return re.sub(r"\s+", " ", text).strip()


def word_error_rate(reference: str, hypothesis: str,
                    normalize: bool = True) -> dict:
    """Levenshtein WER with substitution/insertion/deletion counts."""
    ref = normalize_text(reference).split() if normalize else reference.split()
    hyp = normalize_text(hypothesis).split() if normalize else hypothesis.split()

    if not ref:
        raise ValueError("empty reference")

    # dp[i][j] = (cost, ops) over ref[:i], hyp[:j]
    n, m = len(ref), len(hyp)
    dist = np.zeros((n + 1, m + 1), dtype=np.int32)
    dist[:, 0] = np.arange(n + 1)
    dist[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = dist[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1])
            dist[i, j] = min(sub, dist[i - 1, j] + 1, dist[i, j - 1] + 1)

    # backtrace for op counts
    subs = ins = dels = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dist[i, j] == dist[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            subs += ref[i - 1] != hyp[j - 1]
            i, j = i - 1, j - 1
        elif i > 0 and dist[i, j] == dist[i - 1, j] + 1:
            dels += 1
            i -= 1
        else:
            ins += 1
            j -= 1

    errors = subs + ins + dels
    return {
        "wer": errors / n,
        "substitutions": int(subs),
        "insertions": int(ins),
        "deletions": int(dels),
        "n_ref_words": n,
        "n_hyp_words": m,
    }
