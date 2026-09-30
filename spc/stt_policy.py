"""Streaming transcription policy.

faster-whisper transcribes fixed audio buffers, not streams. This module turns
it into a streaming transcriber with a simple commit policy:

- Audio chunks accumulate in a rolling buffer.
- Every `step_s` seconds of new audio, the whole buffer is re-transcribed.
- Segments that end more than `commit_margin_s` before the buffer's end are
  *committed* (final): their text will not change anymore, and their audio is
  dropped from the buffer. Everything after them is emitted as a *partial*.
- If the buffer grows past `max_buffer_s` without a natural commit point
  (continuous speech), all but the last segment are force-committed to bound
  both memory and per-step inference cost.
- On end-of-stream, everything remaining is committed.

The margin exists because Whisper's transcription of the last ~1 s of a buffer
is unstable (it may cut a word in half). Committing only what is comfortably
in the past trades a little latency for stable, non-flickering finals.

The transcription backend is injected as a callable so this policy is unit
testable without loading a model.
"""
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np


@dataclass
class Segment:
    """A transcribed span, times in seconds of absolute audio time."""
    start: float
    end: float
    text: str


# A transcriber takes (float32 audio, context_prompt) and returns segments
# with times relative to the start of the given audio.
TranscribeFn = Callable[[np.ndarray, Optional[str]], List[Segment]]


class StreamingTranscriber:
    def __init__(
        self,
        transcribe: TranscribeFn,
        sample_rate: int = 16000,
        step_s: float = 1.0,
        commit_margin_s: float = 1.2,
        max_buffer_s: float = 12.0,
        context_words: int = 32,
    ):
        self.transcribe = transcribe
        self.sr = sample_rate
        self.step_s = step_s
        self.commit_margin_s = commit_margin_s
        self.max_buffer_s = max_buffer_s
        self.context_words = context_words

        self._buffer = np.zeros(0, dtype=np.float32)
        self._buffer_start_s = 0.0  # absolute audio time of _buffer[0]
        self._pending_s = 0.0       # new audio since last inference
        self._committed_text: List[str] = []

    @property
    def buffer_duration_s(self) -> float:
        return len(self._buffer) / self.sr

    def _context_prompt(self) -> Optional[str]:
        """Tail of committed text, used to condition the next transcription."""
        if not self._committed_text:
            return None
        words = " ".join(self._committed_text).split()
        return " ".join(words[-self.context_words:])

    def add_audio(
        self, chunk: np.ndarray, eos: bool = False
    ) -> Tuple[List[Segment], Optional[str]]:
        """Feed a float32 audio chunk.

        Returns (committed_segments, partial_text). Both are empty/None unless
        an inference step ran. partial_text may be an empty string, meaning
        "inference ran, nothing pending".
        """
        if len(chunk):
            self._buffer = np.concatenate([self._buffer, chunk])
            self._pending_s += len(chunk) / self.sr

        if not eos and self._pending_s < self.step_s:
            return [], None
        if len(self._buffer) == 0:
            return [], ("" if eos else None)

        self._pending_s = 0.0
        segments = self.transcribe(self._buffer, self._context_prompt())
        buffer_dur = self.buffer_duration_s

        if eos:
            committed = list(segments)
        else:
            cutoff = buffer_dur - self.commit_margin_s
            committed = [s for s in segments if s.end <= cutoff]
            # Force progress on long continuous speech: commit all but the
            # last (still-changing) segment.
            if not committed and buffer_dur > self.max_buffer_s:
                committed = segments[:-1] if len(segments) > 1 else list(segments)

        partial_segments = segments[len(committed):]
        partial_text = " ".join(s.text.strip() for s in partial_segments).strip()

        # Convert to absolute time and trim the buffer past the last commit.
        out: List[Segment] = []
        for seg in committed:
            out.append(Segment(
                start=self._buffer_start_s + seg.start,
                end=self._buffer_start_s + seg.end,
                text=seg.text.strip(),
            ))
        if out:
            trim_s = committed[-1].end
            trim_samples = min(int(trim_s * self.sr), len(self._buffer))
            self._buffer = self._buffer[trim_samples:]
            self._buffer_start_s += trim_samples / self.sr
            self._committed_text.extend(s.text for s in out)

        if eos:
            self._buffer = np.zeros(0, dtype=np.float32)
            partial_text = ""

        return out, partial_text
