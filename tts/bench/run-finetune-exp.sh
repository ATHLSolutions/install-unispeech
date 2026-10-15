#!/usr/bin/env bash
# UniSpeech TTS: LoRA fine-tune experiment on a rented GPU (unispeech_tts finetune/experiment.py,
# docs/huanluyen/phantich.md §6 giai đoạn 0). Trains a LoRA voice with a few settings and compares it with instant
# cloning (speaker similarity, timings, VRAM), then checks the adapter on the CPU path (base ONNX graphs, patched).
# By default the data are SYNTHETIC (the base model reads public sentences with a built-in voice): no person's voice
# ever reaches the rented host. A real voice (DATASET_URL) needs the speaker's consent and should run on our own GPU.
#
#   curl -fsSL https://raw.githubusercontent.com/ATHLSolutions/install-unispeech/main/tts/bench/run-finetune-exp.sh -o /root/tts-ft-exp.sh
#   nohup bash /root/tts-ft-exp.sh > /var/log/tts-ft-exp.log 2>&1 &        # ~20-40 min; survives an SSH drop
#   tail -f /var/log/tts-ft-exp.log
# Variables:
#   CONFIGS            e<epochs>_r<rank>[_all], comma list (default e3_r16,e10_r16,e20_r32)
#   SYNTH_VOICE        built-in voice of the synthetic dataset (default "Mai Anh"); SYNTH_N: first N sentences (0 = all ~100)
#   DATASET_URL        zip with metadata.csv + raw_audio/ of ONE speaker (real voice); needs CONSENT=yes
#   NO_ONNX=1          skip the CPU (ONNX) part
#   EXP_DIR            working folder (default /workspace/tts-ft-exp, else /root/tts-ft-exp)
#   UNISPEECH_FT_VERSION  bundle to use (default tts/finetune/latest)
#   ALLOW_WITH_NODE=1  run even though a speech node is serving on this GPU (it slows its live streams)
# Output: $EXP_DIR/report.md (also printed), results.json, wavs/, results.tar.gz to download.
set -Eeuo pipefail
BASE=${UNISPEECH_INSTALL_BASE:-https://raw.githubusercontent.com/ATHLSolutions/install-unispeech/main/tts}
log() { echo "[tts-ft-exp $(date '+%F %T')] $*"; }
fail() { log "ERROR: $*"; exit 1; }
trap 'fail "line $LINENO: $BASH_COMMAND"' ERR

nvidia-smi --query-gpu=name,memory.used,memory.total,driver_version --format=csv,noheader || fail "nvidia-smi failed: no GPU?"
if pgrep -f 'apps.speech_server serve' >/dev/null && [ "${ALLOW_WITH_NODE:-0}" != 1 ]; then
  fail "a speech node is serving on this machine: training shares its GPU and slows live streams. Drain it first, or ALLOW_WITH_NODE=1"
fi
if [ -n "${DATASET_URL:-}" ] && [ "${CONSENT:-}" != yes ]; then
  fail "DATASET_URL is a person's voice (biometric data): set CONSENT=yes once the speaker agreed, and prefer our own GPU to a rented host"
fi

# ── bundle → /opt/unispeech-ft/<sha> with its own venv (cuda + finetune extras from the bundle's uv.lock) ──
VER=${UNISPEECH_FT_VERSION:-$(curl -fsSL "$BASE/finetune/latest" | tr -d '[:space:]')}
[[ "$VER" =~ ^[0-9a-f]{7,40}$ ]] || fail "bad bundle version '$VER'"
APP=/opt/unispeech-ft/$VER
PY=$APP/venv/bin/python
if [ ! -f "$APP/.installed" ]; then
  log "installing fine-tune bundle $VER"
  if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh
  fi
  PYV=$(curl -fsSL "$BASE/finetune/$VER/PYTHON" | tr -d '[:space:]') || fail "no bundle $VER at $BASE/finetune/$VER"
  bundle=unispeech-finetune-py${PYV/./}.tar.gz
  tmp=$(mktemp -d)
  curl -fsSL "$BASE/finetune/$VER/$bundle" -o "$tmp/$bundle" || fail "cannot download $bundle"
  curl -fsSL "$BASE/finetune/$VER/SHA256SUMS" -o "$tmp/SHA256SUMS"
  (cd "$tmp" && grep " $bundle\$" SHA256SUMS | sha256sum -c --quiet -) || fail "checksum mismatch for $bundle"
  rm -rf "$APP" && mkdir -p "$APP"
  tar -C "$APP" -xzf "$tmp/$bundle"
  rm -rf "$tmp"
  log "venv: Python $PYV, uv sync --frozen --extra cuda --extra finetune"
  (cd "$APP" && UV_PROJECT_ENVIRONMENT="$APP/venv" UV_HTTP_TIMEOUT=300 \
    uv sync -q --frozen --no-dev --extra cuda --extra finetune --no-install-project --no-build --python "$PYV")
  "$PY" -c 'import torch, peft; assert torch.cuda.is_available(), "torch cannot see the GPU"; print("torch", torch.__version__, "peft", peft.__version__, "GPU", torch.cuda.get_device_name(0), "bf16", torch.cuda.is_bf16_supported())' \
    || fail "check failed"
  touch "$APP/.installed"
  for d in /opt/unispeech-ft/*/; do [ "${d%/}" = "$APP" ] || rm -rf "$d"; done
fi

# Same model cache as the speech node when there is one (the base model is then already downloaded).
if [ -z "${HF_HOME:-}" ] && [ -f /etc/environment ]; then HF_HOME=$(sed -n 's/^HF_HOME=//p' /etc/environment | tail -1); fi
if [ -z "${HF_HOME:-}" ]; then
  if [ -d /workspace ]; then HF_HOME=/workspace/.cache/huggingface; else HF_HOME=/root/.cache/huggingface; fi
fi
export HF_HOME PYTHONPATH="$APP/src:$APP"
W=${EXP_DIR:-$([ -d /workspace ] && echo /workspace/tts-ft-exp || echo /root/tts-ft-exp)}
mkdir -p "$W"

configs=${CONFIGS:-e3_r16,e10_r16,e20_r32}
n=$(tr ',' '\n' <<<"$configs" | grep -c .)
need_gb=$(( n * 2 + 6 ))      # merged model ~0.3 GB + patched ONNX copy ~0.4 GB + checkpoints per config, model cache
free_gb=$(df -Pk "$W" | awk 'NR==2 {print int($4 / 1048576)}')
log "disk: ${free_gb} GB free under $W, need about ${need_gb} GB"
[ "$free_gb" -ge "$need_gb" ] || fail "not enough disk"

args=(--work "$W" --configs "$configs")
if [ -n "${DATASET_URL:-}" ]; then
  log "downloading the dataset"
  rm -rf "$W/dataset" && mkdir -p "$W/dataset"
  curl -fsSL "$DATASET_URL" -o "$W/dataset.zip"
  "$PY" - "$W" <<'PY'
import sys, zipfile, pathlib
w = pathlib.Path(sys.argv[1]); ds = w / "dataset"
with zipfile.ZipFile(w / "dataset.zip") as z:
    for m in z.namelist():                                   # no path escapes
        if m.startswith("/") or ".." in pathlib.PurePosixPath(m).parts:
            raise SystemExit(f"unsafe path in zip: {m}")
    z.extractall(ds)
meta = next(ds.rglob("metadata.csv"), None)
if meta is None:
    raise SystemExit("metadata.csv not found in the zip")
if meta.parent != ds:                                        # zip with a top folder: flatten it
    for p in meta.parent.iterdir():
        p.rename(ds / p.name)
print("dataset:", sum(1 for l in open(ds / "metadata.csv", encoding="utf-8") if l.strip()), "lines")
PY
  rm -f "$W/dataset.zip"
  args+=(--dataset-dir "$W/dataset")
else
  rm -rf "$W/dataset"
  args+=(--synthetic-voice "${SYNTH_VOICE:-Mai Anh}" --synthetic-n "${SYNTH_N:-0}")
fi
[ "${NO_ONNX:-0}" = 1 ] && args+=(--no-onnx)

log "experiment $VER: ${args[*]}"
cd "$W"
"$PY" "$APP/finetune/experiment.pyc" "${args[@]}"

if [ -n "${DATASET_URL:-}" ]; then
  rm -rf "$W/dataset" "$W/runs"/*/merged "$W/runs"/*/onnx_update
  log "deleted the dataset and the models trained on it (the speaker's voice); results.tar.gz still holds synthesised clips"
fi
log "done: $W/report.md, download $W/results.tar.gz; remove everything with: rm -rf $W"
