#!/usr/bin/env bash
# UniAgent ASR worker on a vast.ai template (NVIDIA CUDA recommended) or any Linux + NVIDIA driver >= 570.
# Template on-start script (runs as root at every instance start), before the template's own entrypoint.sh,
# which never returns (launch mode Docker ENTRYPOINT, as the vast.ai templates ship):
#   curl -fsSL https://raw.githubusercontent.com/ATHLSolutions/install-unispeech/main/install-worker.sh -o /root/uniasr-install.sh
#   nohup bash /root/uniasr-install.sh >> /var/log/uniasr-install.log 2>&1 &
#   entrypoint.sh
# Variables (template Docker options / rent):
#   UNIASR_GATEWAY_URL    https://asr.uniagent.vn                       (required)
#   UNIASR_NODE_KEY       uan_live_...  per instance, add it when renting (required, never in the template)
#   UNIASR_WORKER_VERSION bundle to install, e.g. 5d84c8f                (default: worker/latest)
# The template's own Python and torch are not used: the worker gets its own venv in /opt/uniasr/<version>/venv
# with the Python minor and the exact package versions of the tested worker image (bundle constraints.txt:
# torch CUDA build, NeMo, numpy, ...). Then the restart loop `uniasr-vast` starts (log /var/log/uniasr-worker.log).
# Already installed (instance restart): only starts the worker again.
set -Eeuo pipefail

BASE=${UNIASR_INSTALL_BASE:-https://raw.githubusercontent.com/ATHLSolutions/install-unispeech/main}
TORCH_INDEX=${UNIASR_TORCH_INDEX:-https://download.pytorch.org/whl/cu128}
LOG=/var/log/uniasr-worker.log
log() { echo "[uniasr-install $(date '+%F %T')] $*"; }
fail() { log "ERROR: $*"; exit 1; }
trap 'fail "line $LINENO: $BASH_COMMAND"' ERR

[ "$(id -u)" -eq 0 ] || fail "run as root (vast.ai on-start does)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || fail "nvidia-smi failed: no GPU?"

# ── worker version ──
VER=${UNIASR_WORKER_VERSION:-$(curl -fsSL "$BASE/worker/latest" | tr -d '[:space:]')}
[[ "$VER" =~ ^[0-9a-f]{7,40}$ ]] || fail "bad worker version '$VER'"
APP=/opt/uniasr/$VER
PY=$APP/venv/bin/python

if [ -f "$APP/.installed" ]; then
  log "worker $VER already installed in $APP"
else
  log "installing worker $VER"
  # System packages: ffmpeg (audio decoding), libsndfile, a C++ compiler for NeMo deps built from source.
  need=()
  command -v ffmpeg >/dev/null || need+=(ffmpeg)
  ldconfig -p | grep libsndfile >/dev/null || need+=(libsndfile1)
  command -v g++ >/dev/null || need+=(build-essential)
  if [ ${#need[@]} -gt 0 ]; then
    log "apt install ${need[*]}"
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq && apt-get install -y -qq --no-install-recommends "${need[@]}" >/dev/null
  fi
  if ! command -v uv >/dev/null; then
    log "installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh
  fi

  # Bundle: bytecode for the image's Python minor, checked against SHA256SUMS.
  PYV=$(curl -fsSL "$BASE/worker/$VER/PYTHON" | tr -d '[:space:]') || fail "no worker $VER at $BASE/worker/$VER"
  bundle=uniasr-worker-py${PYV/./}.tar.gz
  tmp=$(mktemp -d)
  curl -fsSL "$BASE/worker/$VER/$bundle" -o "$tmp/$bundle" || fail "cannot download $bundle"
  curl -fsSL "$BASE/worker/$VER/SHA256SUMS" -o "$tmp/SHA256SUMS"
  (cd "$tmp" && grep " $bundle\$" SHA256SUMS | sha256sum -c --quiet -) || fail "checksum mismatch for $bundle"
  rm -rf "$APP" && mkdir -p "$APP"
  tar -C "$APP" -xzf "$tmp/$bundle"
  rm -rf "$tmp"

  # Own venv, same Python minor and package versions as the worker image (uv fetches that Python if needed).
  log "venv: Python $PYV, $(head -1 "$APP/requirements.txt"), $(wc -l < "$APP/constraints.txt") pinned packages"
  uv venv -q --python "$PYV" "$APP/venv"
  UV_HTTP_TIMEOUT=300 uv pip install --python "$PY" -r "$APP/requirements.txt" -c "$APP/constraints.txt" \
    --extra-index-url "$TORCH_INDEX" --index-strategy unsafe-best-match
  (cd "$APP" && "$PY" - <<'PY'
import torch
assert torch.cuda.is_available(), "torch cannot see the GPU (driver too old for this CUDA build?)"
print("torch", torch.__version__, "GPU", torch.cuda.get_device_name(0))
import nemo.collections.asr, apps.asr_worker.engines.parakeet  # noqa: F401  (the imports the worker needs)
print("nemo.collections.asr OK")
PY
  ) || fail "check failed"
  touch "$APP/.installed"
  # Older versions: free the disk (a venv is several GB).
  for d in /opt/uniasr/*/; do [ "${d%/}" = "$APP" ] || rm -rf "$d"; done
  log "worker $VER installed"
fi
ln -sfn "$APP" /opt/uniasr/current
install -m 755 "$APP/bin/uniasr-vast" /usr/local/bin/uniasr-vast

# ── environment for the worker and for SSH sessions (uniasr-vast reads UNIASR_*/HF_* from it) ──
if [ -z "${HF_HOME:-}" ]; then
  if [ -d /workspace ]; then HF_HOME=/workspace/.cache/huggingface; else HF_HOME=/root/.cache/huggingface; fi
fi
export HF_HOME UNIASR_APP_DIR=/opt/uniasr/current UNIASR_PYTHON=/opt/uniasr/current/venv/bin/python
touch /etc/environment
sed -i -E '/^(UNIASR|HF)_[A-Z0-9_]*=/d' /etc/environment
env | grep -E '^(UNIASR|HF)_[A-Z0-9_]*=' >> /etc/environment
chmod 600 /etc/environment   # holds the node key

# ── (re)start the worker loop ──
if pkill -TERM -f '/usr/local/bin/uniasr-vast' 2>/dev/null; then   # drains running jobs (UNIASR_STOP_GRACE)
  for _ in $(seq 60); do pgrep -f '/usr/local/bin/uniasr-vast' >/dev/null || break; sleep 1; done
fi
[ -n "${UNIASR_NODE_KEY:-}" ] || log "UNIASR_NODE_KEY is empty: add -e UNIASR_NODE_KEY=uan_live_... when renting"
nohup /usr/local/bin/uniasr-vast >> "$LOG" 2>&1 &
log "worker started (pid $!), log: $LOG"
