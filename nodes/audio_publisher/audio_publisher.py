#!/usr/bin/env python3
"""audio_publisher BRAND node.

Simulates a live microphone: loads a wav file, converts it to 16 kHz mono
int16 PCM, and publishes fixed-size chunks to a Redis stream, paced against
the wall clock so downstream nodes see a real-time feed.

Output stream entries (`output_stream` parameter, default "audio"):
    data      int16 PCM bytes (little-endian)
    i         chunk index (ascii int)
    n         number of samples in this chunk
    sr        sample rate (16000)
    t_start_s audio time of the chunk's first sample, seconds (ascii float)
    ts_ns     time.monotonic_ns() immediately before the Redis write
    eos       "0", or "1" on the final empty end-of-stream marker
"""
import gc
import logging
import time

from brand import BRANDNode

from spc.audio import iter_chunks, load_wav_16k_mono, TARGET_SR


class AudioPublisher(BRANDNode):
    def __init__(self):
        super().__init__()
        self.wav_file = self.parameters["wav_file"]
        self.chunk_ms = int(self.parameters.get("chunk_ms", 100))
        self.output_stream = self.parameters.get("output_stream", "audio")
        # realtime=False publishes as fast as possible (useful for tests)
        self.realtime = bool(self.parameters.get("realtime", True))

        self.audio = load_wav_16k_mono(self.wav_file)
        self.chunk_samples = int(TARGET_SR * self.chunk_ms / 1000)
        logging.info(
            "loaded %s: %.2f s, publishing %d ms chunks to '%s' (realtime=%s)",
            self.wav_file, len(self.audio) / TARGET_SR, self.chunk_ms,
            self.output_stream, self.realtime,
        )

    def run(self):
        chunk_ns = int(self.chunk_ms * 1e6)
        t0 = time.monotonic_ns()
        for i, chunk in enumerate(iter_chunks(self.audio, self.chunk_samples)):
            if self.realtime:
                # absolute deadlines avoid cumulative drift
                deadline = t0 + i * chunk_ns
                now = time.monotonic_ns()
                if now < deadline:
                    time.sleep((deadline - now) / 1e9)
            self.r.xadd(self.output_stream, {
                "data": chunk.tobytes(),
                "i": i,
                "n": len(chunk),
                "sr": TARGET_SR,
                "t_start_s": f"{i * self.chunk_ms / 1000:.3f}",
                "ts_ns": time.monotonic_ns(),
                "eos": 0,
            })
        self.r.xadd(self.output_stream, {
            "data": b"",
            "i": i + 1,
            "n": 0,
            "sr": TARGET_SR,
            "t_start_s": f"{len(self.audio) / TARGET_SR:.3f}",
            "ts_ns": time.monotonic_ns(),
            "eos": 1,
        })
        logging.info("end of stream published (%d chunks)", i + 1)
        # Stay alive until the supervisor stops the graph.
        while True:
            time.sleep(1)


if __name__ == "__main__":
    gc.disable()
    node = AudioPublisher()
    node.run()
