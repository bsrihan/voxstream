# speech_to_text

Streaming speech-to-text with [faster-whisper](https://github.com/SYSTRAN/faster-whisper).
Consumes 16 kHz int16 PCM chunks, re-transcribes a rolling buffer every
`step_s` seconds of new audio, and emits:

- **partial** entries: the current, still-revisable tail transcript;
- **final** entries: committed segments whose text will not change (they end
  at least `commit_margin_s` before the newest audio, or the buffer was
  flushed at end-of-stream / `max_buffer_s`).

Committed audio is dropped from the buffer, bounding per-step inference cost.
The tail of committed text conditions the next transcription
(`initial_prompt`) so context is not lost at commit boundaries.

## Parameters

| name | type | default | description |
| --- | --- | --- | --- |
| `input_stream` | str | `audio` | input PCM stream |
| `output_stream` | str | `transcript` | output stream |
| `model` | str | `base.en` | faster-whisper model size |
| `device` | str | `cpu` | `cpu` / `cuda` / `auto` |
| `compute_type` | str | `int8` | ctranslate2 quantization |
| `beam_size` | int | 1 | 1 = greedy (fastest) |
| `step_s` | float | 1.0 | seconds of new audio per inference |
| `commit_margin_s` | float | 1.2 | how far behind the live edge a segment must end to be committed |
| `max_buffer_s` | float | 12.0 | force-commit threshold for continuous speech |
| `log` | str | INFO | log level |

## Output stream entries

`type` (partial/final), `text`, `seg_start_s`/`seg_end_s`/`seg_id` (finals),
`audio_i` + `audio_ts_ns` (newest audio chunk included — latency anchor),
`ts_ns` (write timestamp), `infer_ms`, `eos`.
