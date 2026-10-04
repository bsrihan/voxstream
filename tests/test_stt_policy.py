"""Unit tests for the streaming commit policy with a scripted fake transcriber.

The fake "hears" one word per second of audio, segmented into 2-word
segments, mimicking whisper's (audio -> timestamped segments) interface
deterministically.
"""
import numpy as np
import pytest

from spc.stt_policy import Segment, StreamingTranscriber

SR = 16000
WORDS = "zero one two three four five six seven eight nine".split()


def fake_transcribe(audio: np.ndarray, prompt):
    """One word per second, grouped into 2-word segments.

    Word k spans [k, k+1) s of the *given buffer*. The word identity is
    derived from the sample values so trimming the buffer shifts which words
    appear, like a real transcriber.
    """
    n_words = int(len(audio) / SR)
    if n_words == 0:
        return []
    first_word = int(round(audio[0]))  # samples encode their global second
    segs = []
    for s in range(0, n_words, 2):
        e = min(s + 2, n_words)
        text = " ".join(WORDS[(first_word + k) % 10] for k in range(s, e))
        segs.append(Segment(float(s), float(e), text))
    return segs


def seconds(sec_index: int) -> np.ndarray:
    """1 s of fake audio whose samples encode the global second index."""
    return np.full(SR, float(sec_index), dtype=np.float32)


def make_stt(**kw):
    defaults = dict(sample_rate=SR, step_s=1.0, commit_margin_s=1.2,
                    max_buffer_s=12.0)
    defaults.update(kw)
    return StreamingTranscriber(fake_transcribe, **defaults)


def test_no_inference_before_step():
    stt = make_stt(step_s=2.0)
    committed, partial = stt.add_audio(seconds(0))
    assert committed == [] and partial is None


def test_partial_emitted_before_commit():
    stt = make_stt()
    committed, partial = stt.add_audio(seconds(0))
    # 1 s buffer: segment [0,1) ends within the 1.2 s margin -> not committed
    assert committed == []
    assert partial == "zero"


def test_commit_when_segment_clears_margin():
    stt = make_stt()
    stt.add_audio(seconds(0))
    stt.add_audio(seconds(1))
    stt.add_audio(seconds(2))
    committed, partial = stt.add_audio(seconds(3))
    # 4 s buffer, segment [0,2)="zero one" ends 2 s before end > 1.2 s margin
    assert [c.text for c in committed] == ["zero one"]
    assert committed[0].start == 0.0 and committed[0].end == 2.0
    assert partial == "two three"


def test_buffer_trimmed_after_commit():
    stt = make_stt()
    for k in range(4):
        stt.add_audio(seconds(k))
    assert stt.buffer_duration_s == pytest.approx(2.0)  # 4 s - 2 s committed


def test_commit_times_are_absolute_after_trim():
    stt = make_stt()
    for k in range(4):
        stt.add_audio(seconds(k))  # commits [0,2)
    for k in range(4, 8):
        committed, _ = stt.add_audio(seconds(k))
    # next commit must report absolute audio time, not buffer-relative
    all_committed = committed
    assert all_committed[-1].end > 2.0


def test_eos_flushes_everything():
    stt = make_stt()
    stt.add_audio(seconds(0))
    committed, partial = stt.add_audio(seconds(1), eos=True)
    assert " ".join(c.text for c in committed) == "zero one"
    assert partial == ""
    assert stt.buffer_duration_s == 0.0


def test_eos_on_empty_buffer():
    stt = make_stt()
    committed, partial = stt.add_audio(np.zeros(0, dtype=np.float32), eos=True)
    assert committed == [] and partial == ""


def test_force_commit_bounds_buffer():
    # margin larger than any buffer -> nothing commits naturally
    stt = make_stt(commit_margin_s=100.0, max_buffer_s=5.0)
    total_committed = []
    for k in range(10):
        committed, _ = stt.add_audio(seconds(k))
        total_committed += committed
    assert total_committed, "force-commit never triggered"
    assert stt.buffer_duration_s <= 8.0, "buffer grew without bound"


def test_full_stream_reconstructs_text():
    stt = make_stt()
    out = []
    for k in range(9):
        committed, _ = stt.add_audio(seconds(k))
        out += [c.text for c in committed]
    committed, _ = stt.add_audio(seconds(9), eos=True)
    out += [c.text for c in committed]
    assert " ".join(out).split() == WORDS


def test_context_prompt_from_committed_tail():
    stt = make_stt(context_words=3)
    prompts = []
    original = stt.transcribe

    def spy(audio, prompt):
        prompts.append(prompt)
        return original(audio, prompt)

    stt.transcribe = spy
    for k in range(6):
        stt.add_audio(seconds(k))
    assert prompts[0] is None
    assert any(p is not None and len(p.split()) <= 3 for p in prompts[1:])
