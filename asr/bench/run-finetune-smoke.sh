#!/usr/bin/env bash
# UniSpeech ASR: smoke test of the fine-tuning scripts (unispeech_asr train/) on a rented GPU, with PUBLIC audio
# only (a test sentence recorded for this purpose): checks that NeMo's fine-tuning starts with our settings,
# trains, saves a .nemo, and that the local evaluation runs. Not a measure of quality: the model learns the one
# sentence it sees. Customer recordings are never trained on rented hosts.
# Runs where asr/install-worker.sh installed the worker (its venv has NeMo 2.4.0); needs ~14 GB of free GPU memory.
#
#   curl -fsSL https://raw.githubusercontent.com/ATHLSolutions/install-unispeech/main/asr/bench/run-finetune-smoke.sh | bash
#
# Output: the checkpoint, and WER / term recall of the base model and the checkpoint on the validation copies.
set -euo pipefail
BASE=${UNIASR_INSTALL_BASE:-https://raw.githubusercontent.com/ATHLSolutions/install-unispeech/main/asr}
SAMPLE=https://raw.githubusercontent.com/ATHLSolutions/install-unispeech/f0898b3/asr/bench/Recording2.m4a
TEXT="Tôi đã gửi payment qua Zalo rồi. Đặt xe limousine đi Buôn Ma Thuột."
APP=/opt/uniasr/current
PY=$APP/venv/bin/python
[ -x "$PY" ] || { echo "no worker in $APP: run asr/install-worker.sh first" >&2; exit 1; }
if [ -z "${HF_HOME:-}" ] && [ -f /etc/environment ]; then
  HF_HOME=$(sed -n 's/^HF_HOME=//p' /etc/environment | tail -1)
fi
export HF_HOME=${HF_HOME:-/root/.cache/huggingface}

W=${SMOKE_DIR:-/root/ft-smoke}
rm -rf "$W" && mkdir -p "$W/audio"
PYV=$("$PY" -c 'import sys; print(f"{sys.version_info[0]}{sys.version_info[1]}")')
for f in check_tokens eval_local eval_asr; do
  curl -fsSL "$BASE/bench/$f.py$PYV.pyc" -o "$W/$f.pyc" || { echo "no $f for Python $PYV at $BASE/bench" >&2; exit 1; }
done
curl -fsSL "$BASE/bench/finetune.sh" -o "$W/finetune.sh"

# public sample → 16 kHz mono; 32 training copies, 4 validation copies (same sentence)
curl -fsSL "$SAMPLE" -o "$W/sample.m4a"
ffmpeg -nostdin -v error -y -i "$W/sample.m4a" -ac 1 -ar 16000 -c:a pcm_s16le "$W/audio/sample.wav"
"$PY" - "$W" "$TEXT" <<'PY'
import json, sys, soundfile as sf
w, text = sys.argv[1], sys.argv[2]
d = round(sf.info(f"{w}/audio/sample.wav").duration, 3)
row = json.dumps({"audio_filepath": f"{w}/audio/sample.wav", "duration": d, "text": text}, ensure_ascii=False)
open(f"{w}/train_manifest.jsonl", "w", encoding="utf-8").write((row + "\n") * 32)
open(f"{w}/val_manifest.jsonl", "w", encoding="utf-8").write((row + "\n") * 4)
print(f"sample: {d} s, 32 train / 4 val copies")
PY

nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
# NeMo is verbose: drop its INFO/WARNING lines, keep the rest
EPOCHS=${EPOCHS:-3} BATCH=${BATCH:-4} ACCUM=${ACCUM:-1} LR=${LR:-1e-4} PYTHON="$PY" bash "$W/finetune.sh" "$W" 2>&1 \
  | sed -E '/^\[NeMo [IW] /d' | grep -vE '^\s*$|it/s\]$' | tail -40

CKPT=$(find "$W/exp" -name '*.nemo' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
[ -n "$CKPT" ] || { echo "FAILED: no .nemo checkpoint under $W/exp" >&2; exit 1; }
echo "checkpoint: $CKPT ($(du -h "$CKPT" | cut -f1))"
"$PY" "$W/eval_local.pyc" --manifest "$W/val_manifest.jsonl" --model base --model "$CKPT" \
  --terms "payment,Zalo,limousine,Buôn Ma Thuột" 2>&1 | sed -E '/^\[NeMo [IW] /d' | grep -E '^\||WER'
echo "done; remove with: rm -rf $W"
