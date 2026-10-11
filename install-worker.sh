#!/usr/bin/env bash
# Moved to asr/install-worker.sh (the repo now has asr/ and tts/). Kept for vast.ai templates whose on-start
# still downloads this URL; point them to .../main/asr/install-worker.sh, then this file can go.
set -euo pipefail
BASE=${UNIASR_INSTALL_BASE:-https://raw.githubusercontent.com/ATHLSolutions/install-unispeech/main/asr}
exec bash <(curl -fsSL "$BASE/install-worker.sh") "$@"
