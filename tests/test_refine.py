from spc.refine import (TranscriptRefiner, build_user_prompt, clean_output)


def test_clean_output_strips_wrapping_quotes():
    assert clean_output('"hello there"') == "hello there"
    assert clean_output("'hello'") == "hello"


def test_clean_output_strips_labels():
    assert clean_output("Corrected segment: hello") == "hello"
    assert clean_output("corrected: hello") == "hello"


def test_clean_output_keeps_normal_text():
    assert clean_output("A salt pickle tastes fine with ham.") == \
        "A salt pickle tastes fine with ham."


def test_prompt_includes_context_only_when_present():
    assert "Context" not in build_user_prompt("seg", None)
    assert "previous words" in build_user_prompt("seg", "previous words")


def test_refiner_uses_llm_reply():
    refiner = TranscriptRefiner(lambda sys, usr: "corrected text")
    assert refiner.refine("corupted text") == "corrected text"
    assert refiner.full_text == "corrected text"


def test_refiner_accumulates_context():
    seen_prompts = []

    def gen(sys, usr):
        seen_prompts.append(usr)
        return "ok"

    refiner = TranscriptRefiner(gen)
    refiner.refine("first segment")
    refiner.refine("second segment")
    assert "Context" not in seen_prompts[0]
    assert "ok" in seen_prompts[1]  # first refined output is now context


def test_refiner_falls_back_on_empty_llm_output():
    refiner = TranscriptRefiner(lambda sys, usr: "")
    assert refiner.refine("keep me") == "keep me"


def test_refiner_falls_back_on_runaway_output():
    refiner = TranscriptRefiner(lambda sys, usr: "blah " * 200)
    assert refiner.refine("short segment") == "short segment"


def test_refiner_context_window_bounded():
    refiner = TranscriptRefiner(lambda sys, usr: "w " * 50, context_words=10)
    # runaway guard: each refine falls back to raw, but context still bounded
    for i in range(10):
        refiner.refine(f"segment number {i} with several words in it")
    ctx = refiner._context()
    assert len(ctx.split()) <= 10


def test_empty_segment_skips_llm():
    calls = []

    def gen(sys, usr):
        calls.append(usr)
        return "x"

    refiner = TranscriptRefiner(gen)
    assert refiner.refine("   ") == ""
    assert calls == []
