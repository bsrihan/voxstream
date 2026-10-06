#!/usr/bin/env python3
"""text_refiner BRAND node.

Consumes committed (final) transcript segments and refines them with a small
local instruction-tuned LLM (default Qwen2.5-0.5B-Instruct): fixes ASR errors
using context, restores punctuation, but does not paraphrase. Partial entries
are ignored — refining text that may still change would waste the latency
budget.

Output stream entries (`output_stream` parameter, default "refined"):
    seg_id       id of the refined final segment
    raw_text     ASR text in
    text         refined text out
    audio_ts_ns  latency anchor propagated from the audio stream
    stt_ts_ns    write timestamp of the consumed transcript entry
    ts_ns        time.monotonic_ns() immediately before this Redis write
    infer_ms     LLM generation time for this segment
    eos          "0"; the final marker ("1") carries full_text
"""
import gc
import logging
import time

from brand import BRANDNode

from spc.refine import TranscriptRefiner


class TextRefiner(BRANDNode):
    def __init__(self):
        super().__init__()
        p = self.parameters
        self.input_stream = p.get("input_stream", "transcript")
        self.output_stream = p.get("output_stream", "refined")
        self.model_name = p.get("model", "Qwen/Qwen2.5-0.5B-Instruct")
        self.device = p.get("device", "auto")
        self.max_new_tokens = int(p.get("max_new_tokens", 128))
        # passthrough mode: emit raw text unrefined (for ablation runs)
        self.enabled = bool(p.get("enabled", True))

        if self.enabled:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            if self.device == "auto":
                self.device = "mps" if torch.backends.mps.is_available() else \
                    ("cuda" if torch.cuda.is_available() else "cpu")
            t = time.monotonic()
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            ).to(self.device).eval()
            logging.info("loaded %s on %s in %.1f s", self.model_name,
                         self.device, time.monotonic() - t)
            t = time.monotonic()
            self._generate("You are a helpful assistant.", "Say OK.")
            logging.info("warmup generation in %.2f s", time.monotonic() - t)
        else:
            logging.info("refinement disabled: passthrough mode")

        self.refiner = TranscriptRefiner(self._generate)

    def _generate(self, system_prompt: str, user_prompt: str) -> str:
        import torch
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}]
        # Template to text, then tokenize explicitly: the return type of
        # apply_chat_template(return_tensors=...) changed across transformers
        # versions, this path is stable.
        text = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        n_input = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(out[0, n_input:],
                                     skip_special_tokens=True)

    def run(self):
        last_id = "0-0"
        while True:
            streams = self.r.xread({self.input_stream: last_id},
                                   block=1000, count=10)
            if not streams:
                continue
            _, entries = streams[0]
            for entry_id, entry in entries:
                last_id = entry_id
                if entry[b"type"] != b"final":
                    continue
                if entry[b"eos"] == b"1":
                    self.r.xadd(self.output_stream, {
                        "seg_id": "",
                        "raw_text": "",
                        "text": "",
                        "full_text": self.refiner.full_text,
                        "audio_ts_ns": int(entry[b"audio_ts_ns"]),
                        "stt_ts_ns": int(entry[b"ts_ns"]),
                        "ts_ns": time.monotonic_ns(),
                        "infer_ms": "0.0",
                        "eos": 1,
                    })
                    logging.info("end of stream; full text: %s",
                                 self.refiner.full_text)
                    continue

                raw = entry[b"text"].decode()
                t_inf = time.monotonic()
                if self.enabled:
                    refined = self.refiner.refine(raw)
                else:
                    refined = raw
                    self.refiner._refined.append(raw)
                infer_ms = (time.monotonic() - t_inf) * 1000

                self.r.xadd(self.output_stream, {
                    "seg_id": int(entry[b"seg_id"]),
                    "raw_text": raw,
                    "text": refined,
                    "audio_ts_ns": int(entry[b"audio_ts_ns"]),
                    "stt_ts_ns": int(entry[b"ts_ns"]),
                    "ts_ns": time.monotonic_ns(),
                    "infer_ms": f"{infer_ms:.1f}",
                    "eos": 0,
                })
                logging.info("refined #%s: %s", entry[b"seg_id"].decode(),
                             refined)


if __name__ == "__main__":
    gc.disable()
    node = TextRefiner()
    node.run()
