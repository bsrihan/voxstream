#!/usr/bin/env python3
"""Stub speech_to_text node for integration tests.

Speaks the same stream schema as the real speech_to_text node but requires no
model: emits a partial every `partial_every` chunks and one fixed final
segment at end-of-stream.
"""
import time

from brand import BRANDNode


class StubSTT(BRANDNode):
    def __init__(self):
        super().__init__()
        self.input_stream = self.parameters.get("input_stream", "audio")
        self.output_stream = self.parameters.get("output_stream", "transcript")
        self.final_text = self.parameters.get("final_text", "stub final text")
        self.partial_every = int(self.parameters.get("partial_every", 5))

    def run(self):
        last_id = "0-0"
        n = 0
        while True:
            streams = self.r.xread({self.input_stream: last_id},
                                   block=1000, count=100)
            if not streams:
                continue
            for entry_id, entry in streams[0][1]:
                last_id = entry_id
                base = {
                    "audio_i": int(entry[b"i"]),
                    "audio_ts_ns": int(entry[b"ts_ns"]),
                    "infer_ms": "0.0",
                }
                if entry[b"eos"] == b"1":
                    self.r.xadd(self.output_stream, {
                        **base, "type": "final", "text": self.final_text,
                        "seg_start_s": "0.00", "seg_end_s": "1.00",
                        "seg_id": 0, "ts_ns": time.monotonic_ns(), "eos": 0,
                    })
                    self.r.xadd(self.output_stream, {
                        **base, "type": "final", "text": "",
                        "seg_start_s": "", "seg_end_s": "", "seg_id": "",
                        "ts_ns": time.monotonic_ns(), "eos": 1,
                    })
                    continue
                n += 1
                if n % self.partial_every == 0:
                    self.r.xadd(self.output_stream, {
                        **base, "type": "partial", "text": f"partial {n}",
                        "seg_start_s": "", "seg_end_s": "", "seg_id": "",
                        "ts_ns": time.monotonic_ns(), "eos": 0,
                    })


if __name__ == "__main__":
    StubSTT().run()
