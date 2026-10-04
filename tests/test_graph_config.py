"""Sanity checks on the committed graph YAML and repo layout: every node in
the graph exists on disk with source + Makefile, and the streams wire up."""
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
GRAPH = REPO / "graphs" / "speech_pipeline" / "speech_pipeline.yaml"


def load_nodes():
    with open(GRAPH) as f:
        return yaml.safe_load(f)["nodes"]


def test_graph_parses_and_has_three_nodes():
    nodes = load_nodes()
    assert [n["name"] for n in nodes] == [
        "audio_publisher", "speech_to_text", "text_refiner"]


def test_nicknames_unique():
    nodes = load_nodes()
    nicknames = [n["nickname"] for n in nodes]
    assert len(nicknames) == len(set(nicknames))


def test_node_sources_exist():
    for node in load_nodes():
        d = REPO / "nodes" / node["name"]
        assert (d / f"{node['name']}.py").exists(), f"missing source in {d}"
        assert (d / "Makefile").exists(), f"missing Makefile in {d}"
        assert (d / "README.md").exists(), f"missing README in {d}"


def test_streams_wire_up():
    nodes = {n["nickname"]: n["parameters"] for n in load_nodes()}
    assert nodes["audio_publisher"]["output_stream"] == \
        nodes["speech_to_text"]["input_stream"]
    assert nodes["speech_to_text"]["output_stream"] == \
        nodes["text_refiner"]["input_stream"]


def test_no_linux_only_options():
    """run_priority/cpu_affinity invoke chrt/taskset; keep the graph portable."""
    for node in load_nodes():
        assert "run_priority" not in node
        assert "cpu_affinity" not in node


def test_load_graph_overrides():
    from spc.brand_run import load_graph
    graph = load_graph("data/speech_1.wav",
                       speech_to_text={"model": "small.en"})
    params = {n["nickname"]: n["parameters"] for n in graph["nodes"]}
    assert params["audio_publisher"]["wav_file"] == "data/speech_1.wav"
    assert params["speech_to_text"]["model"] == "small.en"
    # untouched parameters survive
    assert params["text_refiner"]["input_stream"] == "transcript"
