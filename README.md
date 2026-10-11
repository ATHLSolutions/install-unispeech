# install-unispeech

Script cài đặt nhanh cho **UniAgent ASR** trên máy GPU (Vast.ai), dùng model **NVIDIA Parakeet-CTC-0.6B-Vietnamese** (`nvidia/parakeet-ctc-0.6b-Vietnamese`).

| Script | Dùng để |
|---|---|
| [`install-worker.sh`](install-worker.sh) | Cài và chạy **GPU worker** UniAgent ASR (nhận job từ `asr.uniagent.vn`) khi thuê máy trên Vast.ai (template NVIDIA CUDA) |
| [`setup-parakeet.sh`](setup-parakeet.sh) | Cài NeMo và tạo script `transcribe.py` để thử model trên một file audio |

## GPU worker trên Vast.ai (`install-worker.sh`)

Worker được phát hành dưới dạng **bundle bytecode** trong [`worker/`](worker/) (không có mã nguồn). `worker/latest` là phiên bản mới nhất.

Script **không dùng Python hay torch của template**. Nó tạo venv riêng `/opt/uniasr/<version>/venv` với đúng phiên bản Python và **đúng phiên bản mọi gói** của image worker đã kiểm thử (`constraints.txt` trong bundle: torch 2.8.0+cu128, NeMo 2.4.0, numpy, pyarrow…). Môi trường vì vậy giống hệt máy chạy Docker, không bị lệch phiên bản như khi cài chồng lên template PyTorch.

### Thiết lập template

| Mục | Giá trị |
|---|---|
| Image | Template **NVIDIA CUDA** của Vast.ai (chỉ cần driver NVIDIA ≥ 570, hỗ trợ CUDA 12.8). Template khác cũng chạy được vì script không dùng Python/torch có sẵn |
| Launch mode | **Docker ENTRYPOINT** như template gốc, ô Args để trống. Không đổi sang SSH: mode SSH thay entrypoint của image, mất supervisor, Jupyter, Instance Portal |
| On-start script | xem bên dưới |
| Docker options | `-e UNIASR_GATEWAY_URL=https://asr.uniagent.vn -e UNIASR_WORKER_VERSION=5d84c8f -e UNIASR_BATCH_SIZE=8 -e UNIASR_MAX_JOBS=1` |
| Disk | 40 GB (template + NeMo + model 2,4 GB) |

On-start script: chèn hai dòng **trước** `entrypoint.sh` của template. `entrypoint.sh` khởi động SSH, Jupyter, Instance Portal rồi chạy mãi, nên lệnh đặt sau nó không bao giờ chạy. Script cài đặt chạy nền nên không làm chậm `entrypoint.sh`.

```bash
curl -fsSL https://raw.githubusercontent.com/ATHLSolutions/install-unispeech/main/install-worker.sh -o /root/uniasr-install.sh
nohup bash /root/uniasr-install.sh >> /var/log/uniasr-install.log 2>&1 &
entrypoint.sh
```

Extra filters: giữ bộ lọc của template (`cuda_max_good>=12.8` là bắt buộc), thêm `num_gpus=1 gpu_ram>=12 disk_space>=40 reliability>0.98`.

**Không** đặt `UNIASR_NODE_KEY` trong template. Lúc thuê máy, thêm `-e UNIASR_NODE_KEY=uan_live_...` (tạo ở console → GPU workers → Add worker, mỗi máy một key).

### Script làm gì

1. Kiểm tra GPU (`nvidia-smi`).
2. Cài `ffmpeg`, `libsndfile1`, `build-essential` nếu thiếu, và `uv` nếu chưa có.
3. Tải bundle `worker/<version>/uniasr-worker-py3XX.tar.gz` (phiên bản Python ghi trong `worker/<version>/PYTHON`), kiểm tra với `SHA256SUMS`, giải nén vào `/opt/uniasr/<version>`.
4. Tạo venv bằng `uv` (tự tải Python nếu máy chưa có), cài torch CUDA, NeMo và các gói của worker theo `constraints.txt`.
5. Kiểm tra torch thấy GPU và `import nemo.collections.asr` chạy được. Xoá các phiên bản cũ trong `/opt/uniasr`.
6. Ghi biến `UNIASR_*`, `HF_*` vào `/etc/environment` (quyền 600) để phiên SSH dùng được.
7. Chạy `uniasr-vast`: vòng lặp tự khởi động lại worker khi crash, dừng hẳn khi thiếu key hoặc key bị từ chối.

Lần thuê đầu cài mất vài phút (torch + NeMo khoảng 5 GB, ~194 gói) và tải model 2,4 GB khi worker khởi động. Instance *restart* thì bỏ qua bước cài, chỉ chạy lại worker.

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `UNIASR_GATEWAY_URL` | (bắt buộc) | `https://asr.uniagent.vn` |
| `UNIASR_NODE_KEY` | (bắt buộc) | key của máy, thêm lúc thuê |
| `UNIASR_WORKER_VERSION` | nội dung `worker/latest` (hiện là `5d84c8f`) | nên ghim trong template |
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
