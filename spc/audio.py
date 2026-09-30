"""Audio loading and chunking utilities.

All pipeline audio is 16 kHz mono int16 PCM, the native input format of
Whisper-family models. Files with other sample rates / channel counts are
converted on load.
"""
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SR = 16000


def load_wav_16k_mono(path: str, target_sr: int = TARGET_SR) -> np.ndarray:
    """Load a wav file as 16 kHz mono int16 PCM.

    Stereo is downmixed by averaging channels; other sample rates are
    resampled with a polyphase filter (avoids aliasing, exact rational ratio).
    """
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != target_sr:
        g = gcd(int(sr), int(target_sr))
        audio = resample_poly(audio, target_sr // g, sr // g)
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767.0).astype(np.int16)


def pcm16_to_float32(pcm: np.ndarray) -> np.ndarray:
    """Convert int16 PCM to float32 in [-1, 1] (what faster-whisper expects)."""
    return pcm.astype(np.float32) / 32768.0


def iter_chunks(audio: np.ndarray, chunk_samples: int):
    """Yield consecutive chunks of `chunk_samples`; the last one may be short."""
    if chunk_samples <= 0:
        raise ValueError("chunk_samples must be positive")
    for start in range(0, len(audio), chunk_samples):
        yield audio[start:start + chunk_samples]
