#!/usr/bin/env bash
# UniAgent ASR worker on a vast.ai PyTorch template (or any Linux + NVIDIA + CUDA torch).
# Template on-start script (runs as root at every instance start):
#   curl -fsSL https://raw.githubusercontent.com/ATHLSolutions/install-unispeech/main/install-worker.sh -o /root/uniasr-install.sh && nohup bash /root/uniasr-install.sh >> /var/log/uniasr-install.log 2>&1 &
# Variables (template Docker options / rent):
#   UNIASR_GATEWAY_URL    https://asr.uniagent.vn                       (required)
#   UNIASR_NODE_KEY       uan_live_...  per instance, add it when renting (required, never in the template)
#   UNIASR_WORKER_VERSION bundle to install, e.g. 79c2be3                (default: worker/latest)
#   UNIASR_PYTHON         interpreter with CUDA torch                    (default: /venv/main/bin/python)
# Installs NeMo ASR into that interpreter (torch is kept as is), the worker bytecode bundle into
# /opt/uniasr/<version>, then starts the restart loop `uniasr-vast` (log: /var/log/uniasr-worker.log).
# Already installed (instance restart): only starts the worker again.
set -Eeuo pipefail

BASE=${UNIASR_INSTALL_BASE:-https://raw.githubusercontent.com/ATHLSolutions/install-unispeech/main}
LOG=/var/log/uniasr-worker.log
log() { echo "[uniasr-install $(date '+%F %T')] $*"; }
fail() { log "ERROR: $*"; exit 1; }
trap 'fail "line $LINENO: $BASH_COMMAND"' ERR

[ "$(id -u)" -eq 0 ] || fail "run as root (vast.ai on-start does)"

# ── interpreter with CUDA torch ──
PY=${UNIASR_PYTHON:-/venv/main/bin/python}
[ -x "$PY" ] || PY=$(command -v python3 || true)
[ -n "$PY" ] || fail "no python found: set UNIASR_PYTHON"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || fail "nvidia-smi failed: no GPU?"
read -r PYTAG TORCH_VERSION < <("$PY" - <<'PY'
import sys, torch
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access the GPU")
print(f"py{sys.version_info.major}{sys.version_info.minor}", torch.__version__)
PY
) || fail "$PY: torch missing or cannot see the GPU"
log "python $PY ($PYTAG), torch $TORCH_VERSION"

# ── worker version ──
VER=${UNIASR_WORKER_VERSION:-$(curl -fsSL "$BASE/worker/latest" | tr -d '[:space:]')}
[[ "$VER" =~ ^[0-9a-f]{7,40}$ ]] || fail "bad worker version '$VER'"
APP=/opt/uniasr/$VER

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

  # Bundle: bytecode for this Python version, checked against SHA256SUMS.
  tmp=$(mktemp -d)
  bundle=uniasr-worker-$PYTAG.tar.gz
  curl -fsSL "$BASE/worker/$VER/$bundle" -o "$tmp/$bundle" || fail "no bundle $bundle for worker $VER (supported: see $BASE/worker/$VER/SHA256SUMS)"
  curl -fsSL "$BASE/worker/$VER/SHA256SUMS" -o "$tmp/SHA256SUMS"
  (cd "$tmp" && grep " $bundle\$" SHA256SUMS | sha256sum -c --quiet -) || fail "checksum mismatch for $bundle"
  rm -rf "$APP" && mkdir -p "$APP"
  tar -C "$APP" -xzf "$tmp/$bundle"
  rm -rf "$tmp"

  # Python packages, keeping the template's CUDA build of torch.
  echo "torch==$TORCH_VERSION" > "$APP/constraints.txt"
  if command -v uv >/dev/null; then
    uv pip install --python "$PY" -r "$APP/requirements.txt" -c "$APP/constraints.txt"
  else
    "$PY" -m pip install --no-cache-dir -r "$APP/requirements.txt" -c "$APP/constraints.txt"
  fi
  (cd "$APP" && "$PY" -c "import nemo.collections.asr, apps.asr_worker.engines.parakeet") || fail "import check failed"
  touch "$APP/.installed"
  log "worker $VER installed"
fi
ln -sfn "$APP" /opt/uniasr/current
install -m 755 "$APP/bin/uniasr-vast" /usr/local/bin/uniasr-vast

# ── environment for the worker and for SSH sessions (uniasr-vast reads UNIASR_*/HF_* from it) ──
if [ -z "${HF_HOME:-}" ]; then
  if [ -d /workspace ]; then HF_HOME=/workspace/.cache/huggingface; else HF_HOME=/root/.cache/huggingface; fi
fi
export HF_HOME UNIASR_APP_DIR=/opt/uniasr/current UNIASR_PYTHON=$PY
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
