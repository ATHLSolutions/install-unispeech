#!/usr/bin/env bash
# Fine-tune Parakeet-CTC-0.6B-Vietnamese on reviewed call audio (docs/huanluyen/trienkhai.md 6.2), on the customer's
# GPU or ATHL's own (24 GB, bf16: Ampere or newer). Never on a rented public host: the data are customer calls.
#
#   train/finetune.sh /data/ft1                 # manifests from train/build_manifest.py
#   EPOCHS=30 LR=5e-5 BATCH=8 ACCUM=8 train/finetune.sh /data/ft1
#   PYTHON=/opt/uniasr/current/venv/bin/python   # the worker's venv (NeMo 2.4.0, torch CUDA) by default
#
# Settings measured on the GPU (trienkhai.md 5.4): bf16-mixed (fp16 gives NaN from the first step), batch 8 x 15 s
# = 12.25 GB. The tokenizer is kept (it writes every Latin letter; labels are checked by check_tokens.py first).
# The best checkpoint by val_wer (KEEP=1; weights only) is saved as .nemo under <data>/exp; compare it with the base model on the
# eval split with train/eval_local.py before any release.
set -euo pipefail
DATA=${1:?usage: train/finetune.sh <dir with train_manifest.jsonl and val_manifest.jsonl>}
DATA=$(cd "$DATA" && pwd)
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PYTHON:-/opt/uniasr/current/venv/bin/python}
EPOCHS=${EPOCHS:-20} LR=${LR:-3e-5} BATCH=${BATCH:-8} ACCUM=${ACCUM:-8} KEEP=${KEEP:-1}
NEMO_TAG=v2.4.0
for f in train_manifest.jsonl val_manifest.jsonl; do
  [ -s "$DATA/$f" ] || { echo "$DATA/$f is missing or empty" >&2; exit 1; }
done
# Checkpoints: the best KEEP by val_wer, weights only (a Lightning checkpoint with the AdamW state is ~7 GB):
# ~2.4 GB each plus its .nemo (~2.4 GB). A full disk shows up as torch.save "unexpected pos ...".
need_gb=$(( KEEP * 5 + 5 ))
free_gb=$(df -Pk "$DATA" | awk 'NR==2 {print int($4 / 1048576)}')
echo "disk: ${free_gb} GB free under $DATA, need about ${need_gb} GB"
[ "$free_gb" -ge "$need_gb" ] || { echo "not enough disk for the checkpoints (KEEP=$KEEP)" >&2; exit 1; }
"$PY" -c 'import torch, sys; ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported(); print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none", "bf16:", ok); sys.exit(0 if ok else 1)' \
  || { echo "needs a CUDA GPU with bf16" >&2; exit 1; }

# 1. labels the tokenizer cannot write would teach "⁇" (check_tokens.pyc next to this script: bytecode copy)
CT=$HERE/check_tokens.py; [ -f "$CT" ] || CT=$HERE/check_tokens.pyc
"$PY" "$CT" "$DATA/train_manifest.jsonl" "$DATA/val_manifest.jsonl"

# 2. NeMo's fine-tuning script and config, at the version of the worker
SRC="$DATA/nemo-$NEMO_TAG"
mkdir -p "$SRC/conf"
base=https://raw.githubusercontent.com/NVIDIA/NeMo/$NEMO_TAG/examples/asr
[ -s "$SRC/speech_to_text_finetune.py" ] || curl -fsSL "$base/speech_to_text_finetune.py" -o "$SRC/speech_to_text_finetune.py"
[ -s "$SRC/conf/speech_to_text_finetune.yaml" ] || \
  curl -fsSL "$base/conf/asr_finetune/speech_to_text_finetune.yaml" -o "$SRC/conf/speech_to_text_finetune.yaml"

# 3. the base checkpoint (the worker's Hugging Face cache when there is one)
BASE=${BASE_NEMO:-$("$PY" -c 'from huggingface_hub import hf_hub_download as d; print(d("nvidia/parakeet-ctc-0.6b-Vietnamese", "parakeet-ctc-0.6b-vi.nemo"))')}
echo "base: $BASE"

# 4. train
NAME=ft-$(date +%Y%m%d-%H%M)
"$PY" "$SRC/speech_to_text_finetune.py" \
  --config-path="$SRC/conf" --config-name=speech_to_text_finetune \
  name="$NAME" \
  init_from_nemo_model="$BASE" \
  model.tokenizer.update_tokenizer=false \
  model.train_ds.manifest_filepath="$DATA/train_manifest.jsonl" \
  model.validation_ds.manifest_filepath="$DATA/val_manifest.jsonl" \
  model.train_ds.batch_size="$BATCH" model.validation_ds.batch_size="$BATCH" \
  model.train_ds.min_duration=0.5 model.train_ds.max_duration=20 \
  model.train_ds.num_workers=4 model.validation_ds.num_workers=4 \
  model.optim.lr="$LR" model.optim.sched.warmup_steps=null model.optim.sched.warmup_ratio=0.1 \
  model.optim.sched.min_lr=1e-6 \
  trainer.devices=1 trainer.precision=bf16-mixed \
  trainer.max_epochs="$EPOCHS" trainer.accumulate_grad_batches="$ACCUM" trainer.gradient_clip_val=1.0 \
  exp_manager.exp_dir="$DATA/exp" \
  exp_manager.checkpoint_callback_params.save_top_k="$KEEP" \
  exp_manager.checkpoint_callback_params.save_last=false \
  exp_manager.checkpoint_callback_params.save_weights_only=true

echo
echo "checkpoints (.nemo, best by val_wer):"
find "$DATA/exp/$NAME" -name '*.nemo' -printf '%T@ %p\n' 2>/dev/null | sort -n | cut -d' ' -f2-
echo
echo "next: $PY $HERE/eval_local.py --manifest $DATA/eval_manifest.jsonl --model base --model <one of the .nemo above>"
echo "then delete the audio when done: rm -rf $DATA/audio   (customer recordings)"
