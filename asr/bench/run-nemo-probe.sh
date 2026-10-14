#!/usr/bin/env bash
# UniSpeech ASR: what NeMo can do with the Vietnamese Parakeet model for domain adaptation (context biasing, the
# tokenizer on English words, fp32/bf16/fp16 training numerics and memory, encoder adapters).
# Runs on a machine where asr/install-worker.sh installed the worker. Rent a separate instance for it (the
# template without UNIASR_NODE_KEY installs the worker but does not start it): training loads the model several
# times and needs the whole GPU memory; on a production worker it would compete with recognition.
#
#   curl -fsSL https://raw.githubusercontent.com/ATHLSolutions/install-unispeech/main/asr/bench/run-nemo-probe.sh | bash
#   ... | bash -s -- --only bias --phrases "booking,Đà Lạt" --show-text /root/call.wav
#   ... | bash -s -- --only train --modes bf16 adapter --train-batch 4
#
# Without audio files the public sample tuanta.wav is used and texts are printed. The report is written to
# /root/nemo-probe.json.
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
curl -fsSL "$BASE/bench/nemo_probe.py$PYV.pyc" -o "$tmp/nemo_probe.pyc" \
  || { echo "no probe for Python $PYV at $BASE/bench" >&2; exit 1; }

# Audio: the files among the arguments, else the public sample (and then texts may be shown).
args=() files=0
for a in "$@"; do
  if [ -f "$a" ]; then
    files=$((files + 1))
  elif [[ "$a" =~ \.(wav|mp3|m4a|ogg|opus|flac|aac|webm|mp4)$ ]]; then
    echo "no such audio file: $a" >&2
    exit 1
  fi
  args+=("$a")
done
if [ "$files" -eq 0 ]; then
  curl -fsSL "$SAMPLE" -o "$tmp/sample.wav"
  args=("$tmp/sample.wav" --show-text "${args[@]}")
fi

nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
if pgrep -f 'apps.asr_worker' >/dev/null; then
  echo "WARNING: a uniasr worker is running on this machine: training memory figures will be lower than the GPU allows" >&2
fi
cd "$APP"
# NeMo is verbose: drop its INFO/WARNING lines, keep everything else (errors, tracebacks, results).
PYTHONPATH="$APP" "$PY" "$tmp/nemo_probe.pyc" --tmp "$tmp" --json /root/nemo-probe.json "${args[@]}" 2>&1 \
  | sed -E '/^\[NeMo [IW] /d'
