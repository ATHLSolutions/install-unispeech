#!/usr/bin/env bash
# UniSpeech ASR: GPU latency of one recognition call on short utterances (realtime streaming study, P-S0).
# Runs on a vast.ai machine where asr/install-worker.sh installed the worker; the running worker is not touched
# (the benchmark loads its own copy of the model on the same GPU, about 3 GB of VRAM).
#
#   curl -fsSL https://raw.githubusercontent.com/ATHLSolutions/install-unispeech/main/asr/bench/run-bench-stream.sh | bash
#   ... | bash -s -- --durations 1 2 4 8 15 --batch 1 4 8 --reps 10
#
# Audio: the public sample tuanta.wav (8 s, repeated as needed: timing depends on length, not content).
set -euo pipefail
BASE=${UNIASR_INSTALL_BASE:-https://raw.githubusercontent.com/ATHLSolutions/install-unispeech/main/asr}
SAMPLE=https://raw.githubusercontent.com/ATHLSolutions/install-unispeech/7e5724f/tuanta.wav
APP=/opt/uniasr/current
PY=$APP/venv/bin/python
[ -x "$PY" ] || { echo "no worker in $APP: run asr/install-worker.sh first" >&2; exit 1; }

# Reuse the model the worker already downloaded.
if [ -z "${HF_HOME:-}" ] && [ -f /etc/environment ]; then
  HF_HOME=$(sed -n 's/^HF_HOME=//p' /etc/environment | tail -1)
fi
export HF_HOME=${HF_HOME:-/root/.cache/huggingface}

tmp=$(mktemp -d)
trap 'rm -rf -- "$tmp"' EXIT
PYV=$("$PY" -c 'import sys; print(f"{sys.version_info[0]}{sys.version_info[1]}")')
curl -fsSL "$BASE/bench/bench_stream.py$PYV.pyc" -o "$tmp/bench_stream.pyc" \
  || { echo "no benchmark for Python $PYV at $BASE/bench" >&2; exit 1; }
curl -fsSL "$SAMPLE" -o "$tmp/sample.wav"

nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
cd "$APP"
# NeMo is verbose: drop its INFO/WARNING lines, keep everything else (errors, tracebacks, results).
PYTHONPATH="$APP" "$PY" "$tmp/bench_stream.pyc" "$tmp/sample.wav" --show-diff 5 "$@" 2>&1 | sed -E '/^\[NeMo [IW] /d'
