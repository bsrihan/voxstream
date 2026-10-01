#!/usr/bin/env python3
"""speech_to_text BRAND node.

Consumes 16 kHz int16 PCM chunks from the audio stream and produces rapid
initial transcripts with faster-whisper. Uses the StreamingTranscriber commit
policy (spc.stt_policy): every `step_s` seconds of new audio the rolling
buffer is re-transcribed; segments comfortably in the past are committed as
*final* and everything newer is emitted as a revisable *partial*.

Output stream entries (`output_stream` parameter, default "transcript"):
    type        "partial" | "final"
    text        segment text (partial: current unstable tail)
    seg_start_s / seg_end_s   audio time span (finals only, "" for partials)
    seg_id      running index of final segments ("" for partials)
    audio_i     index of the newest audio chunk included in this inference
    audio_ts_ns publisher write-timestamp of that chunk (latency anchor)
    ts_ns       time.monotonic_ns() immediately before this Redis write
    infer_ms    duration of the whisper call that produced this entry
    eos         "0", or "1" on the final marker after the last commit
"""
import gc
import logging
import time

import numpy as np
from brand import BRANDNode

from spc.stt_policy import Segment, StreamingTranscriber


class SpeechToText(BRANDNode):
    def __init__(self):
        super().__init__()
        p = self.parameters
        self.input_stream = p.get("input_stream", "audio")
        self.output_stream = p.get("output_stream", "transcript")
        self.model_size = p.get("model", "base.en")
        self.device = p.get("device", "cpu")
        self.compute_type = p.get("compute_type", "int8")
        self.beam_size = int(p.get("beam_size", 1))
        step_s = float(p.get("step_s", 1.0))
        commit_margin_s = float(p.get("commit_margin_s", 1.2))
        max_buffer_s = float(p.get("max_buffer_s", 12.0))

        # Import here so the node fails fast with a clear log if the model
        # stack is missing, rather than at module import in tests.
        from faster_whisper import WhisperModel
        t = time.monotonic()
        self.model = WhisperModel(self.model_size, device=self.device,
                                  compute_type=self.compute_type)
        logging.info("loaded whisper '%s' (%s/%s) in %.1f s", self.model_size,
                     self.device, self.compute_type, time.monotonic() - t)

        # Warmup: the first transcribe call pays one-time init costs that
        # would otherwise pollute the first latency measurement.
        t = time.monotonic()
        self._transcribe(np.zeros(16000, dtype=np.float32), None)
        logging.info("warmup transcription in %.2f s", time.monotonic() - t)

        self.stt = StreamingTranscriber(
            self._transcribe, sample_rate=16000, step_s=step_s,
            commit_margin_s=commit_margin_s, max_buffer_s=max_buffer_s)
        self.seg_id = 0

    def _transcribe(self, audio, prompt):
        segments, _ = self.model.transcribe(
            audio,
            language="en",
            beam_size=self.beam_size,
            condition_on_previous_text=False,
            initial_prompt=prompt,
        )
        return [Segment(s.start, s.end, s.text) for s in segments]

    def run(self):
        last_id = "0-0"
        while True:
            streams = self.r.xread({self.input_stream: last_id},
                                   block=1000, count=100)
            if not streams:
                continue
            _, entries = streams[0]
            for entry_id, entry in entries:
                last_id = entry_id
                eos = entry[b"eos"] == b"1"
                chunk = np.frombuffer(entry[b"data"], dtype=np.int16)
                chunk = chunk.astype(np.float32) / 32768.0
                audio_i = int(entry[b"i"])
                audio_ts_ns = int(entry[b"ts_ns"])

                t_inf = time.monotonic()
                committed, partial = self.stt.add_audio(chunk, eos=eos)
                infer_ms = (time.monotonic() - t_inf) * 1000

                if not committed and partial is None and not eos:
                    continue  # no inference ran on this chunk
                base = {
                    "audio_i": audio_i,
                    "audio_ts_ns": audio_ts_ns,
                    "infer_ms": f"{infer_ms:.1f}",
                    "eos": 0,
                }
                for seg in committed:
                    self.r.xadd(self.output_stream, {
                        **base,
                        "type": "final",
                        "text": seg.text,
                        "seg_start_s": f"{seg.start:.2f}",
                        "seg_end_s": f"{seg.end:.2f}",
                        "seg_id": self.seg_id,
                        "ts_ns": time.monotonic_ns(),
                    })
                    logging.info("final #%d [%.1f-%.1fs]: %s", self.seg_id,
                                 seg.start, seg.end, seg.text)
                    self.seg_id += 1
                if partial is not None and not eos:
                    self.r.xadd(self.output_stream, {
                        **base,
                        "type": "partial",
                        "text": partial,
                        "seg_start_s": "",
                        "seg_end_s": "",
                        "seg_id": "",
                        "ts_ns": time.monotonic_ns(),
                    })
                if eos:
                    self.r.xadd(self.output_stream, {
                        **base,
                        "type": "final",
                        "text": "",
                        "seg_start_s": "",
                        "seg_end_s": "",
                        "seg_id": "",
                        "ts_ns": time.monotonic_ns(),
                        "eos": 1,
                    })
                    logging.info("end of stream: %d final segments",
                                 self.seg_id)


if __name__ == "__main__":
    gc.disable()
    node = SpeechToText()
    node.run()
