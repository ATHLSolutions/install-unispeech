#!/usr/bin/env bash
# UniAgent TTS speech node on a vast.ai template (NVIDIA CUDA recommended) or any Linux + NVIDIA driver >= 570.
# Template on-start script (runs as root at every instance start), before the template's own entrypoint.sh,
# which never returns (launch mode Docker ENTRYPOINT, as the vast.ai templates ship):
#   curl -fsSL https://raw.githubusercontent.com/ATHLSolutions/install-unispeech/main/tts/install-worker.sh -o /root/unispeech-install.sh
#   nohup bash /root/unispeech-install.sh >> /var/log/unispeech-install.log 2>&1 &
#   entrypoint.sh
# The gateway calls the node: the template needs -p 8000:8000 (the node reports http://$PUBLIC_IPADDR:$VAST_TCP_PORT_8000).
# Variables (template Docker options / rent):
#   UNISPEECH_GATEWAY_URL     https://speech.uniagent.vn                     (required)
#   UNISPEECH_NODE_KEY        usn_live_...  per instance, add it when renting (required, never in the template)
#   UNISPEECH_NODE_ALLOW_IPS  <gateway public IP>/32                         (strongly advised: plain HTTP)
#   UNISPEECH_WORKER_VERSION  bundle to install, e.g. 6a53901                (default: tts/worker/latest)
# The template's own Python and torch are not used: the node gets its own venv in /opt/unispeech/<version>/venv,
# synced from the bundle's uv.lock exactly like the GPU image (Python 3.12, torch 2.8.0+cu128, ...).
# Then the restart loop `unispeech-vast` starts (log /var/log/unispeech-worker.log).
# Already installed (instance restart): only starts the node again.
set -Eeuo pipefail

BASE=${UNISPEECH_INSTALL_BASE:-https://raw.githubusercontent.com/ATHLSolutions/install-unispeech/main/tts}
LOG=/var/log/unispeech-worker.log
log() { echo "[unispeech-install $(date '+%F %T')] $*"; }
fail() { log "ERROR: $*"; exit 1; }
trap 'fail "line $LINENO: $BASH_COMMAND"' ERR

[ "$(id -u)" -eq 0 ] || fail "run as root (vast.ai on-start does)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || fail "nvidia-smi failed: no GPU?"

# ── node version ──
VER=${UNISPEECH_WORKER_VERSION:-$(curl -fsSL "$BASE/worker/latest" | tr -d '[:space:]')}
[[ "$VER" =~ ^[0-9a-f]{7,40}$ ]] || fail "bad worker version '$VER'"
APP=/opt/unispeech/$VER
PY=$APP/venv/bin/python

if [ -f "$APP/.installed" ]; then
  log "speech node $VER already installed in $APP"
else
  log "installing speech node $VER"
  if ! command -v uv >/dev/null; then
    log "installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh
  fi

  # Bundle: bytecode for the image's Python minor, checked against SHA256SUMS.
  PYV=$(curl -fsSL "$BASE/worker/$VER/PYTHON" | tr -d '[:space:]') || fail "no speech node $VER at $BASE/worker/$VER"
  bundle=unispeech-worker-py${PYV/./}.tar.gz
  tmp=$(mktemp -d)
  curl -fsSL "$BASE/worker/$VER/$bundle" -o "$tmp/$bundle" || fail "cannot download $bundle"
  curl -fsSL "$BASE/worker/$VER/SHA256SUMS" -o "$tmp/SHA256SUMS"
  (cd "$tmp" && grep " $bundle\$" SHA256SUMS | sha256sum -c --quiet -) || fail "checksum mismatch for $bundle"
  rm -rf "$APP" && mkdir -p "$APP"
  tar -C "$APP" -xzf "$tmp/$bundle"
  rm -rf "$tmp"

  # Own venv from uv.lock, as docker/Dockerfile.gpu does (uv fetches Python $PYV if the template has none).
  log "venv: Python $PYV, uv sync --frozen --extra cuda --extra server"
  (cd "$APP" && UV_PROJECT_ENVIRONMENT="$APP/venv" UV_HTTP_TIMEOUT=300 \
    uv sync -q --frozen --no-dev --extra cuda --extra server --no-install-project --no-build --python "$PYV")
  (cd "$APP" && PYTHONPATH="$APP/src:$APP" "$PY" - <<'PY'
import torch
assert torch.cuda.is_available(), "torch cannot see the GPU (driver too old for this CUDA build?)"
print("torch", torch.__version__, "GPU", torch.cuda.get_device_name(0))
import apps.speech_server.__main__, apps.speech_server.worker  # noqa: F401  (what `serve` imports for a node)
print("apps.speech_server.worker OK")
PY
  ) || fail "check failed"
  touch "$APP/.installed"
  # Older versions: free the disk (a venv is several GB).
  for d in /opt/unispeech/*/; do [ "${d%/}" = "$APP" ] || rm -rf "$d"; done
  log "speech node $VER installed"
fi
ln -sfn "$APP" /opt/unispeech/current
install -m 755 "$APP/bin/unispeech-vast" /usr/local/bin/unispeech-vast

# ── environment for the node and for SSH sessions (unispeech-vast reads these from it) ──
if [ -z "${HF_HOME:-}" ]; then
  if [ -d /workspace ]; then HF_HOME=/workspace/.cache/huggingface; else HF_HOME=/root/.cache/huggingface; fi
fi
export HF_HOME UNISPEECH_APP_DIR=/opt/unispeech/current UNISPEECH_PYTHON=/opt/unispeech/current/venv/bin/python
vars='^((UNISPEECH|HF)_[A-Z0-9_]*|PUBLIC_IPADDR|VAST_TCP_PORT_[0-9]+)='
touch /etc/environment
sed -i -E "/$vars/d" /etc/environment
env | grep -E "$vars" >> /etc/environment
chmod 600 /etc/environment   # holds the node key

[ -n "${VAST_TCP_PORT_8000:-}" ] || [ -n "${UNISPEECH_NODE_PUBLIC_URL:-}" ] \
  || log "no VAST_TCP_PORT_8000: add -p 8000:8000 to the template Docker options (the gateway calls the node)"
[ -n "${UNISPEECH_NODE_ALLOW_IPS:-}" ] || log "UNISPEECH_NODE_ALLOW_IPS is empty: anyone may try /internal/v1 (plain HTTP)"

# ── (re)start the node loop ──
if pkill -TERM -f '/usr/local/bin/unispeech-vast' 2>/dev/null; then   # logs out, finishes its streams (30 s)
  for _ in $(seq 60); do pgrep -f '/usr/local/bin/unispeech-vast' >/dev/null || break; sleep 1; done
fi
[ -n "${UNISPEECH_NODE_KEY:-}" ] || log "UNISPEECH_NODE_KEY is empty: add -e UNISPEECH_NODE_KEY=usn_live_... when renting"
nohup /usr/local/bin/unispeech-vast >> "$LOG" 2>&1 &
log "speech node started (pid $!), log: $LOG"
