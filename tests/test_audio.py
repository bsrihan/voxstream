import numpy as np
import pytest
import soundfile as sf

from spc.audio import iter_chunks, load_wav_16k_mono, pcm16_to_float32


@pytest.fixture
def tone_wav(tmp_path):
    """1 s 440 Hz tone factory with configurable rate/channels."""
    def make(sr=16000, channels=1, duration_s=1.0):
        t = np.arange(int(sr * duration_s)) / sr
        x = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        if channels == 2:
            x = np.stack([x, x], axis=1)
        path = tmp_path / f"tone_{sr}_{channels}.wav"
        sf.write(path, x, sr)
        return path
    return make


def test_load_16k_mono_passthrough(tone_wav):
    audio = load_wav_16k_mono(tone_wav(sr=16000, channels=1))
    assert audio.dtype == np.int16
    assert len(audio) == 16000


def test_load_resamples_44k_stereo(tone_wav):
    audio = load_wav_16k_mono(tone_wav(sr=44100, channels=2))
    assert audio.dtype == np.int16
    # exactly 1 s at 16 kHz after rational resampling
    assert abs(len(audio) - 16000) <= 1


def test_load_preserves_signal_energy(tone_wav):
    audio = pcm16_to_float32(load_wav_16k_mono(tone_wav(sr=44100, channels=2)))
    # 0.5-amplitude sine has RMS 0.5/sqrt(2) ~ 0.354
    assert 0.3 < np.sqrt((audio ** 2).mean()) < 0.4


def test_pcm16_float32_round_trip():
    pcm = np.array([-32768, -1, 0, 1, 32767], dtype=np.int16)
    f = pcm16_to_float32(pcm)
    assert f.dtype == np.float32
    assert f.min() >= -1.0 and f.max() <= 1.0
    assert f[2] == 0.0


def test_iter_chunks_covers_everything():
    audio = np.arange(10, dtype=np.int16)
    chunks = list(iter_chunks(audio, 3))
    assert [len(c) for c in chunks] == [3, 3, 3, 1]
    assert np.array_equal(np.concatenate(chunks), audio)


def test_iter_chunks_rejects_bad_size():
    with pytest.raises(ValueError):
        list(iter_chunks(np.zeros(4, dtype=np.int16), 0))
