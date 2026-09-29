#!/usr/bin/env bash
# One-shot environment setup for the real-time speech pipeline.
# Requires: python3 (>=3.10), redis-server on PATH, git, make.
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v redis-server >/dev/null; then
    echo "redis-server not found. Install it first:" >&2
    echo "  macOS:  brew install redis" >&2
    echo "  Ubuntu: sudo apt-get install redis-server" >&2
    exit 1
fi

# BRAND core (supervisor + brand python library)
if [ ! -d third_party/brand ]; then
    git clone https://github.com/brandbci/brand third_party/brand
fi

python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e third_party/brand/lib/python   # `brand` library
.venv/bin/pip install -e .                              # `spc` library

# Build node "binaries" (executable copies of the python sources)
make

echo
echo "Setup complete. Activate with: source .venv/bin/activate"
