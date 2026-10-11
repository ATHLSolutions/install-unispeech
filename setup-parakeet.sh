#!/usr/bin/env bash
set -Eeuo pipefail

echo "========================================"
echo " UniAgent Speech - Parakeet Vietnamese"
echo "========================================"

# 1. Check GPU
nvidia-smi

# 2. Activate Vast Python environment
source /venv/main/bin/activate

# 3. Check Python and PyTorch
python --version

python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise SystemExit("ERROR: PyTorch cannot access the GPU")

print("GPU:", torch.cuda.get_device_name(0))
PY

# 4. Install NVIDIA NeMo ASR
# Pin and validate package versions before production deployment.
uv pip install --python /venv/main/bin/python \
    "nemo_toolkit[asr]"

# 5. Create workspace
mkdir -p /workspace/uniagent-speech
cd /workspace/uniagent-speech

# 6. Create transcription test program
cat > transcribe.py <<'PY'
import sys
import torch
import nemo.collections.asr as nemo_asr

MODEL_NAME = "nvidia/parakeet-ctc-0.6b-Vietnamese"

if len(sys.argv) != 2:
    print("Usage: python transcribe.py /path/to/audio.wav")
    sys.exit(1)

audio_path = sys.argv[1]

print("Loading Parakeet Vietnamese...")
model = nemo_asr.models.ASRModel.from_pretrained(
    model_name=MODEL_NAME
)

model.eval()

print("Transcribing:", audio_path)

with torch.inference_mode():
    results = model.transcribe(
        [audio_path],
        timestamps=True
    )

result = results[0]

print("\n===== TRANSCRIPT =====")
print(result.text)

if getattr(result, "timestamp", None):
    print("\n===== WORD TIMESTAMPS =====")
    print(result.timestamp.get("word", []))
PY

echo "========================================"
echo "Installation completed."
echo "Test with:"
echo "python /workspace/uniagent-speech/transcribe.py /workspace/call.wav"
echo "========================================"