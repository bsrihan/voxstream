"""End-to-end integration test of the BRAND graph mechanics.

Runs the real supervisor (which starts its own redis-server) with the real
audio_publisher node, a stub speech_to_text node (no model download), and the
real text_refiner node in passthrough mode (no model download). Verifies the
full chain: wav file -> paced audio stream -> transcript stream -> refined
stream with end-of-stream marker, plus latency anchors on every hop.

Requires redis-server on PATH and third_party/brand cloned (setup.sh does
both). Skipped otherwise.
"""
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from spc import brand_run
from spc.brand_run import BRAND_DIR, REPO_ROOT, start_supervisor

pytestmark = pytest.mark.integration

PORT = 28123

requires_stack = pytest.mark.skipif(
    shutil.which("redis-server") is None or not BRAND_DIR.exists(),
    reason="needs redis-server and third_party/brand (run setup.sh)",
)


@pytest.fixture(scope="module")
def built_nodes():
    subprocess.run(["make"], cwd=REPO_ROOT, check=True, capture_output=True)
    subprocess.run(["make"], cwd=REPO_ROOT / "tests" / "fixtures" / "nodes" / "stub_stt",
                   check=True, capture_output=True)


@pytest.fixture(scope="module")
def short_wav(tmp_path_factory):
    """2 s of noise-ish audio, 16 kHz mono (contents don't matter to the stub)."""
    path = tmp_path_factory.mktemp("audio") / "short.wav"
    rng = np.random.default_rng(0)
    sf.write(path, 0.1 * rng.standard_normal(32000).astype(np.float32), 16000)
    return path


def stub_graph(wav_path: Path) -> dict:
    return {
        "graph_name": "integration_stub",
        "nodes": [
            {
                "name": "audio_publisher",
                "nickname": "audio_publisher",
                "module": "../..",
                "parameters": {
                    "log": "INFO",
                    "wav_file": str(wav_path),
                    "chunk_ms": 100,
                    "output_stream": "audio",
                    "realtime": True,
                },
            },
            {
                "name": "stub_stt",
                "nickname": "stub_stt",
                "module": "../../tests/fixtures",
                "parameters": {
                    "log": "INFO",
                    "input_stream": "audio",
                    "output_stream": "transcript",
                    "final_text": "stub final text",
                    "partial_every": 5,
                },
            },
            {
                "name": "text_refiner",
                "nickname": "text_refiner",
                "module": "../..",
                "parameters": {
                    "log": "INFO",
                    "input_stream": "transcript",
                    "output_stream": "refined",
                    "enabled": False,  # passthrough: no model download
                },
            },
        ],
    }


@requires_stack
def test_full_graph_run(built_nodes, short_wav):
    sup = start_supervisor(port=PORT)
    try:
        sup.start_graph(stub_graph(short_wav))
        sup.wait_for_status("running", timeout_s=30)

        # wait for the end-of-stream marker on the refined stream
        deadline = time.monotonic() + 30
        eos_seen = False
        while time.monotonic() < deadline and not eos_seen:
            for _, entry in sup.redis.xrange("refined", "-", "+"):
                if entry[b"eos"] == b"1":
                    eos_seen = True
            time.sleep(0.2)
        assert eos_seen, "refined stream never reached end-of-stream"

        # audio stream: 20 chunks of 100 ms + eos marker, paced in real time
        audio = sup.redis.xrange("audio", "-", "+")
        assert len(audio) == 21
        assert audio[-1][1][b"eos"] == b"1"
        pcm = np.frombuffer(audio[0][1][b"data"], dtype=np.int16)
        assert len(pcm) == 1600
        ts = [int(e[b"ts_ns"]) for _, e in audio[:-1]]
        total_s = (ts[-1] - ts[0]) / 1e9
        assert 1.5 < total_s < 3.5, f"pacing off: 1.9 s expected, got {total_s:.2f} s"

        # transcript stream: partials plus the stub final
        transcript = sup.redis.xrange("transcript", "-", "+")
        types = [e[b"type"].decode() for _, e in transcript]
        assert "partial" in types and "final" in types

        # refined stream: passthrough of the stub final + full_text on eos
        refined = sup.redis.xrange("refined", "-", "+")
        finals = [e for _, e in refined if e[b"eos"] == b"0"]
        assert finals[0][b"text"] == b"stub final text"
        eos_entry = [e for _, e in refined if e[b"eos"] == b"1"][0]
        assert eos_entry[b"full_text"] == b"stub final text"

        # latency anchors propagate and increase along the chain
        f = finals[0]
        assert int(f[b"audio_ts_ns"]) < int(f[b"stt_ts_ns"]) < int(f[b"ts_ns"])

        sup.stop_graph()
        sup.wait_for_status("stopped/not initialized", timeout_s=15)
    finally:
        sup.shutdown()


@requires_stack
def test_node_states_published(built_nodes, short_wav):
    """Nodes must publish their state stream per the BRAND node spec."""
    sup = start_supervisor(port=PORT + 1)
    try:
        sup.start_graph(stub_graph(short_wav))
        sup.wait_for_status("running", timeout_s=30)
        time.sleep(1)
        for nickname in ("audio_publisher", "stub_stt", "text_refiner"):
            state = sup.redis.xrange(f"{nickname}_state", "-", "+")
            assert state, f"{nickname} published no state"
            assert state[0][1][b"status"] == b"initialized"
    finally:
        sup.shutdown()
