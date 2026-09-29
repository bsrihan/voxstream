# voxstream

**Live speech in, polished text out. ~400 ms to first words, nothing leaves your laptop.**

Streaming speech-to-text with LLM post-correction, running fully locally on a laptop
CPU. A small Whisper model produces fast partial transcripts while audio is still
arriving, and a small instruction-tuned LLM cleans up each finalized segment, all
wired together as a [BRAND](https://github.com/brandbci/brand) graph of independent
processes that talk over Redis streams.

Work in progress.
