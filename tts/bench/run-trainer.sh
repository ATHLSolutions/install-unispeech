#!/usr/bin/env bash
# UniSpeech TTS: training machine. Installs the fine-tune bundle (own venv, cuda + finetune extras from its
# uv.lock) and runs the agent finetune/trainer.py in a restart loop: it pulls LoRA training jobs created on the
# console (Huấn luyện), trains, evaluates, uploads the adapter package (unispeech_tts docs/huanluyen/huanluyen-console.md).
#
#   curl -fsSL https://raw.githubusercontent.com/ATHLSolutions/install-unispeech/main/tts/bench/run-trainer.sh -o /root/tts-trainer.sh
#   UNISPEECH_GATEWAY_URL=https://speech.uniagent.vn UNISPEECH_TRAINER_KEY=ust_live_... \
#     nohup bash /root/tts-trainer.sh > /var/log/tts-trainer.log 2>&1 &
# Variables:
#   UNISPEECH_GATEWAY_URL   gateway (required)
#   UNISPEECH_TRAINER_KEY   ust_live_… from console › Huấn luyện › Máy huấn luyện (required)
#   UNISPEECH_FT_VERSION    bundle to use (default tts/finetune/latest)
#   UNISPEECH_TRAINER_WORK  working folder (default /workspace/tts-trainer)
# The machine only calls out (gateway + object storage). A job's dataset is deleted from it when the job ends.
# Exit code 2 of the agent (key refused, missing variables) stops the loop; anything else restarts it.
set -Eeuo pipefail
BASE=${UNISPEECH_INSTALL_BASE:-https://raw.githubusercontent.com/ATHLSolutions/install-unispeech/main/tts}
log() { echo "[tts-trainer $(date '+%F %T')] $*"; }
fail() { log "ERROR: $*"; exit 1; }
trap 'fail "line $LINENO: $BASH_COMMAND"' ERR

[ -n "${UNISPEECH_GATEWAY_URL:-}" ] || fail "set UNISPEECH_GATEWAY_URL (e.g. https://speech.uniagent.vn)"
[[ "${UNISPEECH_TRAINER_KEY:-}" == ust_live_* ]] || fail "set UNISPEECH_TRAINER_KEY (ust_live_…, console › Huấn luyện › Máy huấn luyện)"
nvidia-smi --query-gpu=name,memory.used,memory.total,driver_version --format=csv,noheader || fail "nvidia-smi failed: no GPU?"

# ── bundle → /opt/unispeech-ft/<sha> with its own venv ──
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
  "$PY" -c 'import torch, peft; assert torch.cuda.is_available(), "torch cannot see the GPU"; print("torch", torch.__version__, "GPU", torch.cuda.get_device_name(0), "bf16", torch.cuda.is_bf16_supported())' \
    || fail "check failed"
  touch "$APP/.installed"
  for d in /opt/unispeech-ft/*/; do [ "${d%/}" = "$APP" ] || rm -rf "$d"; done
fi

# Same model cache as a speech node on this machine, if any.
if [ -z "${HF_HOME:-}" ] && [ -f /etc/environment ]; then HF_HOME=$(sed -n 's/^HF_HOME=//p' /etc/environment | tail -1); fi
if [ -z "${HF_HOME:-}" ]; then
  if [ -d /workspace ]; then HF_HOME=/workspace/.cache/huggingface; else HF_HOME=/root/.cache/huggingface; fi
fi
export HF_HOME PYTHONPATH="$APP/src:$APP" UNISPEECH_GATEWAY_URL UNISPEECH_TRAINER_KEY

# ── agent loop ──
trap - ERR
set +e
delay=10
while true; do
  start=$(date +%s)
  log "agent $VER starting"
  "$PY" "$APP/finetune/trainer.pyc"
  code=$?
  if [ "$code" -eq 2 ]; then log "agent exited with 2 (key refused or misconfigured): not restarting"; exit 2; fi
  [ $(( $(date +%s) - start )) -gt 600 ] && delay=10
  log "agent exited with $code; restarting in ${delay}s"
  sleep "$delay"
  delay=$(( delay * 2 > 300 ? 300 : delay * 2 ))
done
