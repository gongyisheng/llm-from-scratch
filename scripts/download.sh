#!/bin/bash
# Download model checkpoint from HuggingFace
# Usage: bash scripts/download.sh <org/model-name>
# Example: bash scripts/download.sh Qwen/Qwen3-0.6B

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

HF_REPO="${1:-Qwen/Qwen3-0.6B}"

if [[ "$HF_REPO" != */* ]]; then
    echo "Error: model must be in 'org/model' format (e.g., Qwen/Qwen3-0.6B)"
    exit 1
fi

MODEL_NAME="${HF_REPO##*/}"
TARGET_DIR="$REPO_ROOT/checkpoints/$MODEL_NAME"

pip install -q huggingface_hub
huggingface-cli download "$HF_REPO" --local-dir "$TARGET_DIR"

echo "Downloaded to $TARGET_DIR"
