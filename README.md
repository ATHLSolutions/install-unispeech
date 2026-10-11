# install-unispeech

Script cài đặt nhanh **NVIDIA Parakeet-CTC-0.6B-Vietnamese** (`nvidia/parakeet-ctc-0.6b-vi`) để nhận dạng tiếng nói tiếng Việt (ASR) cho UniAgent Speech.

## Yêu cầu

- Máy Linux có GPU NVIDIA và driver đã cài (`nvidia-smi` chạy được)
- Python venv tại `/venv/main` có sẵn PyTorch hỗ trợ CUDA (mặc định trên template PyTorch của [Vast.ai](https://vast.ai))
- `uv` đã cài sẵn
- Kết nối Internet để tải package và model từ Hugging Face

> Nếu môi trường của bạn không phải Vast.ai, sửa đường dẫn `/venv/main` trong `setup-parakeet.sh` cho phù hợp.

## Cài đặt

```bash
git clone https://github.com/ATHLSolutions/install-unispeech.git
cd install-unispeech
chmod +x setup-parakeet.sh
./setup-parakeet.sh
```

Script sẽ:

1. Kiểm tra GPU bằng `nvidia-smi`
2. Kích hoạt venv `/venv/main`
3. Kiểm tra phiên bản Python, PyTorch và CUDA (dừng lại nếu PyTorch không thấy GPU)
4. Cài `nemo_toolkit[asr]` bằng `uv pip`
5. Tạo thư mục làm việc `/workspace/uniagent-speech`
6. Tạo script thử nghiệm `transcribe.py`

## Sử dụng

```bash
source /venv/main/bin/activate
python /workspace/uniagent-speech/transcribe.py /workspace/call.wav
```

Lần chạy đầu tiên model sẽ được tải từ Hugging Face và lưu cache. Kết quả in ra gồm:

- `TRANSCRIPT`: văn bản nhận dạng
- `WORD TIMESTAMPS`: mốc thời gian theo từng từ

### Định dạng audio

Model hoạt động tốt nhất với file WAV mono, 16 kHz. Có thể chuyển đổi bằng `ffmpeg`:

```bash
ffmpeg -i input.mp3 -ac 1 -ar 16000 /workspace/call.wav
```

## Lưu ý

- Script cài phiên bản `nemo_toolkit` mới nhất. Trước khi đưa vào production, hãy pin phiên bản cụ thể đã được kiểm thử.
- Thông tin model: https://huggingface.co/nvidia/parakeet-ctc-0.6b-vi
