"""Transcript refinement via a small instruction-tuned LLM.

The refiner receives committed (final) ASR segments one at a time and returns
a corrected version, using the previously refined text as context. The LLM
backend is injected as a callable so the logic is unit testable without
loading a model.

Prompt design notes:
- The model is told to fix ASR errors only, not to paraphrase; unnecessary
  rewrites hurt WER against the reference.
- Filler words ("uh", "um") are explicitly preserved: the provided reference
  transcripts keep them, so removing them counts as deletions.
- Output is constrained to the corrected text alone; `clean_output` strips
  the chat wrappers models sometimes add anyway.
"""
import re
from typing import Callable, List, Optional

SYSTEM_PROMPT = (
    "You are a transcription corrector. You receive one segment of an "
    "automatic speech transcript at a time, possibly with surrounding "
    "context. Correct misrecognized words, spelling, and punctuation using "
    "the context. Do NOT paraphrase, summarize, or drop words. Keep filler "
    "words like 'uh' and 'um'. If the segment is already correct, return it "
    "unchanged. Reply with the corrected segment text only."
)

# Generator takes (system_prompt, user_prompt) and returns the model reply.
GenerateFn = Callable[[str, str], str]


def build_user_prompt(segment: str, context: Optional[str]) -> str:
    if context:
        return (
            f"Context (already corrected, do not repeat it):\n{context}\n\n"
            f"Segment to correct:\n{segment}"
        )
    return f"Segment to correct:\n{segment}"


def strip_context_echo(refined: str, context: Optional[str],
                       min_words: int = 3) -> str:
    """Remove a leading repetition of the context from the model output.

    Small LLMs sometimes prepend the provided context despite being told not
    to. If a suffix of the context (>= min_words words, compared
    case/punctuation-insensitively) is a prefix of the output, drop it.
    """
    if not context:
        return refined

    def key(w: str) -> str:
        return re.sub(r"[^a-z0-9']+", "", w.lower())

    ctx = context.split()
    out = refined.split()
    best = 0
    for k in range(1, min(len(ctx), len(out)) + 1):
        # a short overlap only counts as an echo if it is the whole context
        if k < min_words and k != len(ctx):
            continue
        if [key(w) for w in ctx[-k:]] == [key(w) for w in out[:k]]:
            best = k
    return " ".join(out[best:]) if best else refined


def clean_output(text: str) -> str:
    """Strip quotes/labels an LLM may wrap around the corrected segment."""
    text = text.strip()
    # drop a leading label like "Corrected segment:"
    text = re.sub(r"^(corrected( segment)?|segment|output)\s*:\s*", "", text,
                  flags=re.IGNORECASE)
    # drop symmetric wrapping quotes
    if len(text) >= 2 and text[0] in "\"'\u201c" and text[-1] in "\"'\u201d":
        text = text[1:-1]
    return text.strip()


class TranscriptRefiner:
    def __init__(self, generate: GenerateFn, context_words: int = 60):
        self.generate = generate
        self.context_words = context_words
        self._refined: List[str] = []

    @property
    def full_text(self) -> str:
        return " ".join(self._refined).strip()

    def _context(self) -> Optional[str]:
        if not self._refined:
            return None
        words = self.full_text.split()
        return " ".join(words[-self.context_words:])

    def refine(self, segment: str) -> str:
        segment = segment.strip()
        if not segment:
            return segment
        context = self._context()
        reply = self.generate(SYSTEM_PROMPT,
                              build_user_prompt(segment, context))
        refined = strip_context_echo(clean_output(reply), context)
        # Guard against degenerate LLM outputs (empty, or runaway generation):
        # fall back to the raw ASR segment rather than corrupting the document.
        if not refined or len(refined) > 4 * max(len(segment), 20):
            refined = segment
        self._refined.append(refined)
        return refined
