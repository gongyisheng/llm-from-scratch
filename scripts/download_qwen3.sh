#!/bin/bash
# Download Qwen3 model checkpoint
# Usage: bash scripts/download_qwen3.sh [Qwen3-0.6B|Qwen3-1.7B|Qwen3-4B]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

MODEL_NAME="${1:-Qwen3-0.6B}"

case "$MODEL_NAME" in
    Qwen3-0.6B) HF_REPO="Qwen/Qwen3-0.6B" ;;
    Qwen3-1.7B) HF_REPO="Qwen/Qwen3-1.7B" ;;
    Qwen3-4B)   HF_REPO="Qwen/Qwen3-4B" ;;
    *)
        echo "Unknown model: $MODEL_NAME"
        echo "Supported: Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B"
        exit 1
        ;;
esac

TARGET_DIR="$REPO_ROOT/checkpoints/$MODEL_NAME"

pip install huggingface_hub
huggingface-cli download "$HF_REPO" --local-dir "$TARGET_DIR"

echo "Downloaded to $TARGET_DIR"
