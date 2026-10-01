# audio_publisher

Simulates a live microphone. Loads a wav file, converts it to 16 kHz mono
int16 PCM, and publishes fixed-size chunks to a Redis stream, paced in real
time (a 100 ms chunk is published every 100 ms of wall clock).

## Parameters

| name | type | default | description |
| --- | --- | --- | --- |
| `wav_file` | str | required | path to the input wav (relative paths resolve from the BRAND core dir the supervisor runs in) |
| `chunk_ms` | int | 100 | chunk duration in milliseconds |
| `output_stream` | str | `audio` | Redis stream to publish to |
| `realtime` | bool | true | pace chunks against the wall clock; false = publish as fast as possible |
| `log` | str | INFO | log level |

## Output stream entries

`data` (int16 PCM bytes), `i` (chunk index), `n` (samples), `sr`,
`t_start_s` (audio time of first sample), `ts_ns` (monotonic ns before the
Redis write, for latency analysis), `eos` (1 on the final empty marker).
