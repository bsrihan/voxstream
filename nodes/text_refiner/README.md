# text_refiner

Semantic refinement and error correction of committed transcript segments
with a small local instruction-tuned LLM (default
[Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)).
Each final segment is corrected in the context of the previously refined
text; partial (still-changing) transcripts are ignored by design. Degenerate
LLM outputs (empty / runaway) fall back to the raw ASR segment.

## Parameters

| name | type | default | description |
| --- | --- | --- | --- |
| `input_stream` | str | `transcript` | stream of transcript entries |
| `output_stream` | str | `refined` | output stream |
| `model` | str | `Qwen/Qwen2.5-0.5B-Instruct` | HF causal LM |
| `device` | str | `auto` | auto → mps / cuda / cpu |
| `max_new_tokens` | int | 128 | generation cap per segment |
| `enabled` | bool | true | false = passthrough (ablation: measures pipeline without refinement) |
| `log` | str | INFO | log level |

## Output stream entries

`seg_id`, `raw_text`, `text` (refined), `audio_ts_ns` / `stt_ts_ns` / `ts_ns`
(latency anchors), `infer_ms`, `eos` (final marker carries `full_text`).
