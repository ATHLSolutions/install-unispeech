# install-unispeech

Script cài đặt nhanh cho **UniAgent ASR** trên máy GPU (Vast.ai), dùng model **NVIDIA Parakeet-CTC-0.6B-Vietnamese** (`nvidia/parakeet-ctc-0.6b-Vietnamese`).

| Script | Dùng để |
|---|---|
| [`install-worker.sh`](install-worker.sh) | Cài và chạy **GPU worker** UniAgent ASR (nhận job từ `asr.uniagent.vn`) khi thuê máy bằng template PyTorch của Vast.ai |
| [`setup-parakeet.sh`](setup-parakeet.sh) | Cài NeMo và tạo script `transcribe.py` để thử model trên một file audio |

## GPU worker trên template PyTorch của Vast.ai (`install-worker.sh`)

Worker được phát hành dưới dạng **bundle bytecode** trong [`worker/`](worker/) (không có mã nguồn), mỗi phiên bản Python một file. `worker/latest` là phiên bản mới nhất.

### Thiết lập template

| Mục | Giá trị |
|---|---|
| Image | Template **PyTorch** của Vast.ai (có `/venv/main` với PyTorch CUDA, `uv`). Python 3.10, 3.11 hoặc 3.12 |
| Launch mode | SSH (hoặc Jupyter + SSH) |
| On-start script | xem bên dưới |
| Docker options | `-e UNIASR_GATEWAY_URL=https://asr.uniagent.vn -e UNIASR_WORKER_VERSION=<sha> -e UNIASR_BATCH_SIZE=8 -e UNIASR_MAX_JOBS=1` |
| Disk | 40 GB (template + NeMo + model 2,4 GB) |

On-start script:

```bash
curl -fsSL https://raw.githubusercontent.com/ATHLSolutions/install-unispeech/main/install-worker.sh -o /root/uniasr-install.sh && nohup bash /root/uniasr-install.sh >> /var/log/uniasr-install.log 2>&1 &
```

**Không** đặt `UNIASR_NODE_KEY` trong template. Lúc thuê máy, thêm `-e UNIASR_NODE_KEY=uan_live_...` (tạo ở console → GPU workers → Add worker, mỗi máy một key).

### Script làm gì

1. Kiểm tra GPU (`nvidia-smi`) và PyTorch thấy CUDA (interpreter `UNIASR_PYTHON`, mặc định `/venv/main/bin/python`).
2. Cài `ffmpeg`, `libsndfile1`, `build-essential` nếu thiếu.
3. Tải bundle `worker/<version>/uniasr-worker-py3XX.tar.gz` đúng phiên bản Python, kiểm tra với `SHA256SUMS`, giải nén vào `/opt/uniasr/<version>`.
4. Cài `nemo_toolkit[asr]` (đã pin) và các gói của worker, **giữ nguyên torch** của template.
5. Ghi biến `UNIASR_*`, `HF_*` vào `/etc/environment` (quyền 600) để phiên SSH dùng được.
6. Chạy `uniasr-vast`: vòng lặp tự khởi động lại worker khi crash, dừng hẳn khi thiếu key hoặc key bị từ chối.

Lần thuê đầu cài mất vài phút (NeMo) và tải model khi worker khởi động. Instance *restart* thì bỏ qua bước cài, chỉ chạy lại worker.

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `UNIASR_GATEWAY_URL` | (bắt buộc) | `https://asr.uniagent.vn` |
| `UNIASR_NODE_KEY` | (bắt buộc) | key của máy, thêm lúc thuê |
| `UNIASR_WORKER_VERSION` | nội dung `worker/latest` (hiện là `e8a0e46`) | nên ghim trong template |
| `UNIASR_PYTHON` | `/venv/main/bin/python` | interpreter có PyTorch CUDA |
| `HF_HOME` | `/workspace/.cache/huggingface` | cache model |

### Theo dõi

```bash
tail -f /var/log/uniasr-install.log    # cài đặt
tail -f /var/log/uniasr-worker.log     # worker: chờ "logged in to https://asr.uniagent.vn"
```

Đổi key qua SSH: `UNIASR_NODE_KEY=uan_live_... bash /root/uniasr-install.sh` (lần restart instance sau vẫn dùng key trong Docker options).

## Thử model trên một file (`setup-parakeet.sh`)

### Yêu cầu

- Máy Linux có GPU NVIDIA và driver đã cài (`nvidia-smi` chạy được)
- Python venv tại `/venv/main` có sẵn PyTorch hỗ trợ CUDA (mặc định trên template PyTorch của [Vast.ai](https://vast.ai))
- `uv` đã cài sẵn
- Kết nối Internet để tải package và model từ Hugging Face

> Nếu môi trường của bạn không phải Vast.ai, sửa đường dẫn `/venv/main` trong `setup-parakeet.sh` cho phù hợp.

### Cài đặt

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

### Sử dụng

```bash
source /venv/main/bin/activate
python /workspace/uniagent-speech/transcribe.py /workspace/call.wav
```

Lần chạy đầu tiên model sẽ được tải từ Hugging Face và lưu cache. Kết quả in ra gồm:

- `TRANSCRIPT`: văn bản nhận dạng
- `WORD TIMESTAMPS`: mốc thời gian theo từng từ

#### Định dạng audio

Model hoạt động tốt nhất với file WAV mono, 16 kHz. Có thể chuyển đổi bằng `ffmpeg`:

```bash
ffmpeg -i input.mp3 -ac 1 -ar 16000 /workspace/call.wav
```

### Lưu ý

- Script cài phiên bản `nemo_toolkit` mới nhất. Trước khi đưa vào production, hãy pin phiên bản cụ thể đã được kiểm thử.
- Thông tin model: https://huggingface.co/nvidia/parakeet-ctc-0.6b-Vietnamese
