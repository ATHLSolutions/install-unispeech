# install-unispeech

Script cài đặt nhanh **GPU worker** của UniAgent Speech trên máy GPU thuê (Vast.ai, template NVIDIA CUDA). Mỗi dịch vụ một thư mục:

| Thư mục | Dịch vụ | Script | Bundle |
|---|---|---|---|
| [`asr/`](asr/) | **ASR**: nhận dạng giọng nói, kéo job từ `asr.uniagent.vn` (NVIDIA Parakeet-CTC-0.6B-Vietnamese) | [`asr/install-worker.sh`](asr/install-worker.sh) | `asr/worker/<sha>/`, `asr/worker/latest` |
| [`tts/`](tts/) | **TTS**: speech node, gateway `speech.uniagent.vn` gọi vào (UniSpeech v3 Turbo) | [`tts/install-worker.sh`](tts/install-worker.sh) | `tts/worker/<sha>/`, `tts/worker/latest` |

Worker phát hành dưới dạng **bundle bytecode** (không có mã nguồn), tạo bằng `scripts/build_worker_bundle.sh` của repo `unispeech_asr` / `unispeech_tts`. Script cài đặt không dùng Python hay torch của template: mỗi worker có venv riêng (`/opt/uniasr/<sha>`, `/opt/unispeech/<sha>`) với đúng phiên bản gói của image đã kiểm thử.

> Template ASR cũ tải `main/install-worker.sh` (thư mục gốc) phải đổi on-start sang `main/asr/install-worker.sh`: file ở thư mục gốc đã bị xoá.

## ASR: GPU worker trên Vast.ai (`asr/install-worker.sh`)

Worker được phát hành dưới dạng **bundle bytecode** trong [`asr/worker/`](asr/worker/) (không có mã nguồn). `asr/worker/latest` là phiên bản mới nhất.

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
curl -fsSL https://raw.githubusercontent.com/ATHLSolutions/install-unispeech/main/asr/install-worker.sh -o /root/uniasr-install.sh
nohup bash /root/uniasr-install.sh >> /var/log/uniasr-install.log 2>&1 &
entrypoint.sh
```

Extra filters: giữ bộ lọc của template (`cuda_max_good>=12.8` là bắt buộc), thêm `num_gpus=1 gpu_ram>=12 disk_space>=40 reliability>0.98`.

**Không** đặt `UNIASR_NODE_KEY` trong template. Lúc thuê máy, thêm `-e UNIASR_NODE_KEY=uan_live_...` (tạo ở console → GPU workers → Add worker, mỗi máy một key).

### Script làm gì

1. Kiểm tra GPU (`nvidia-smi`).
2. Cài `ffmpeg`, `libsndfile1`, `build-essential` nếu thiếu, và `uv` nếu chưa có.
3. Tải bundle `asr/worker/<version>/uniasr-worker-py3XX.tar.gz` (phiên bản Python ghi trong `asr/worker/<version>/PYTHON`), kiểm tra với `SHA256SUMS`, giải nén vào `/opt/uniasr/<version>`.
4. Tạo venv bằng `uv` (tự tải Python nếu máy chưa có), cài torch CUDA, NeMo và các gói của worker theo `constraints.txt`.
5. Kiểm tra torch thấy GPU và `import nemo.collections.asr` chạy được. Xoá các phiên bản cũ trong `/opt/uniasr`.
6. Ghi biến `UNIASR_*`, `HF_*` vào `/etc/environment` (quyền 600) để phiên SSH dùng được.
7. Chạy `uniasr-vast`: vòng lặp tự khởi động lại worker khi crash, dừng hẳn khi thiếu key hoặc key bị từ chối.

Lần thuê đầu cài mất vài phút (torch + NeMo khoảng 5 GB, ~194 gói) và tải model 2,4 GB khi worker khởi động. Instance *restart* thì bỏ qua bước cài, chỉ chạy lại worker.

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `UNIASR_GATEWAY_URL` | (bắt buộc) | `https://asr.uniagent.vn` |
| `UNIASR_NODE_KEY` | (bắt buộc) | key của máy, thêm lúc thuê |
| `UNIASR_WORKER_VERSION` | nội dung `asr/worker/latest` (hiện là `5d84c8f`) | nên ghim trong template |
| `HF_HOME` | `/workspace/.cache/huggingface` | cache model |

### Theo dõi

```bash
tail -f /var/log/uniasr-install.log    # cài đặt
tail -f /var/log/uniasr-worker.log     # worker: chờ "logged in to https://asr.uniagent.vn"
```

Đổi key qua SSH: `UNIASR_NODE_KEY=uan_live_... bash /root/uniasr-install.sh` (lần restart instance sau vẫn dùng key trong Docker options).

## TTS: speech node trên Vast.ai (`tts/install-worker.sh`)

Khác ASR: gateway **gọi vào** node, nên template phải mở cổng `-p 8000:8000`. Node tự báo URL `http://$PUBLIC_IPADDR:$VAST_TCP_PORT_8000` khi đăng nhập. Đây là HTTP thường qua internet: đặt `UNISPEECH_NODE_ALLOW_IPS` là IP công khai của gateway.

Venv được tạo bằng `uv sync --frozen` từ `uv.lock` trong bundle, y như image `unispeech-speech:gpu` (Python 3.12, torch 2.8.0+cu128).

### Thiết lập template

| Mục | Giá trị |
|---|---|
| Image | Template **NVIDIA CUDA** của Vast.ai (driver NVIDIA ≥ 570, CUDA 12.8) |
| Launch mode | **Docker ENTRYPOINT** như template gốc, ô Args để trống |
| On-start script | xem bên dưới |
| Docker options | `-p 8000:8000 -e UNISPEECH_GATEWAY_URL=https://speech.uniagent.vn -e UNISPEECH_NODE_ALLOW_IPS=<IP gateway>/32 -e UNISPEECH_WORKER_VERSION=<sha>` |
| Disk | 30 GB (venv torch ~6 GB + model) |

**Không** thêm cổng 8000 vào `PORTAL_CONFIG`: Caddy của Instance Portal gắn xác thực riêng và chặn gateway.

```bash
curl -fsSL https://raw.githubusercontent.com/ATHLSolutions/install-unispeech/main/tts/install-worker.sh -o /root/unispeech-install.sh
nohup bash /root/unispeech-install.sh >> /var/log/unispeech-install.log 2>&1 &
entrypoint.sh
```

Extra filters: giữ bộ lọc của template (`cuda_max_good>=12.8`), thêm `num_gpus=1 gpu_ram>=8 disk_space>=30 reliability>0.98`.

Lúc thuê máy, thêm `-e UNISPEECH_NODE_KEY=usn_live_...` (console → Speech nodes → Thêm node, mỗi máy một key).

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `UNISPEECH_GATEWAY_URL` | (bắt buộc) | `https://speech.uniagent.vn` |
| `UNISPEECH_NODE_KEY` | (bắt buộc) | key của node, thêm lúc thuê |
| `UNISPEECH_NODE_ALLOW_IPS` | (trống) | IP gateway `/32`, nên đặt |
| `UNISPEECH_WORKER_VERSION` | nội dung `tts/worker/latest` | nên ghim trong template |
| `HF_HOME` | `/workspace/.cache/huggingface` | cache model |

Theo dõi:

```bash
tail -f /var/log/unispeech-install.log   # cài đặt
tail -f /var/log/unispeech-worker.log    # node: chờ "logged in to https://speech.uniagent.vn"
```

Node thoát mã `2` (thiếu key, thiếu `-p 8000:8000`) thì `unispeech-vast` không khởi động lại, chỉ ghi lý do vào log. Đổi key qua SSH: `UNISPEECH_NODE_KEY=usn_live_... bash /root/unispeech-install.sh`.
