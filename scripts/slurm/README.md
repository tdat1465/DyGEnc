# Chạy DyGEnc AGQA trên server: data/model/environment trong RAM

Bộ script này dành riêng cho **DyGEnc**, không gọi code/config/checkpoint GRN.
Nó chạy toàn bộ các QA có grounding không rỗng, không áp dụng giới hạn chiều dài
50/10 như config AGQA gốc. `train`/`test` vẫn là hai split gốc của DyGEnc.

## File và nơi lưu

- `train_agqa_full_ram.slurm`: bắt đầu một run mới.
- `resume_agqa_full_ram.slurm`: tiếp tục `last.pth` của chính runner này.
- `agqa_full_job.sh`: dựng venv, cài dependency, tải data/model, preprocess và train
  trong **cùng một allocation**. Không submit preprocess thành job riêng: RAM bị
  dọn khi job kết thúc.
- `stage_agqa.py`: chuẩn hóa năm raw input, kiểm tra mapping và ghi SHA-256.
- `src/server_train.py`: runner riêng có resume; không gọi `train.py` cũ.

| Nơi lưu | Nội dung |
| --- | --- |
| Persistent disk | Source code; `last.pth`, `best.pth`; log/metadata nhỏ; hai token và mapping `ENG.txt` nhỏ |
| Private tmpfs dưới `RAM_BASE` | Venv/dependency, pip/temp/cache, archive tải về, raw AGQA, dữ liệu preprocess, ModernBERT và Llama snapshots |
| GPU VRAM | Trọng số/tensor phục vụ tính toán; tmpfs không thay thế GPU VRAM |

Không có fallback tải data/model xuống disk. Source đang chạy được snapshot vào
RAM để tránh bị thay đổi giữa job. `HOME` không bị đổi; các cache/temp được chuyển
hướng riêng. Trainer không chạy Trackio/W&B hoặc tạo Hugging Face Space.

## 1. Đưa code đã sửa lên server

Dùng repo `tdat1465/DyGEnc` với tag `kaggle-ram-v1` để lấy đúng bản có các script
và runner mới. Clone nguyên upstream `linukc/DyGEnc` sẽ chưa có các bổ sung này.
Cell clone dùng trên Kaggle nằm trong `notebooks/kaggle_clone.ipynb`. Nếu upload
source thủ công, cần cả `src/`, `scripts/`, `requirements-server.txt`; không chỉ
copy hai file `.slurm` vì chúng cần helper và runner.

Ví dụ sau dùng mặc định tương tự script GRN đã cung cấp:

```bash
export PERSIST_ROOT=/media02/lnthanh03
cd "$PERSIST_ROOT/code/DyGEnc"
mkdir -p logs "$PERSIST_ROOT/secrets"
```

Nếu bạn dùng `/media02/lnthanh03/DatHa`, đổi `PERSIST_ROOT` thành đường dẫn đó và
đứng trong checkout DyGEnc thực tế trước khi submit. `SOURCE_REPO` mặc định lấy
`SLURM_SUBMIT_DIR`; có thể export đường dẫn tuyệt đối khác.

## 2. Chuẩn bị ba file nhỏ, không tải dataset/model xuống disk

```text
$PERSIST_ROOT/secrets/kaggle.json   # Kaggle legacy API credentials
$PERSIST_ROOT/secrets/hf_token     # một dòng Hugging Face read token
$PERSIST_ROOT/secrets/ENG.txt      # mapping AGQA JSON, nếu dataset thiếu file này
```

Hugging Face account phải được phép truy cập
[meta-llama/Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B).
Lưu token bằng editor/upload file, không viết token trong lệnh `sbatch` hoặc
commit lên Git. Nếu dùng editor Windows, token một dòng với CRLF cũng được xử lý.

```bash
chmod 600 "$PERSIST_ROOT/secrets/kaggle.json" "$PERSIST_ROOT/secrets/hf_token"
```

Dataset mặc định là `tdat1465/agqa-balanced`. Nó phải cung cấp:

```text
train_balanced.txt
test_balanced.txt
AGQA_train_stsgs.pkl
AGQA_test_stsgs.pkl
```

Nếu archive có duy nhất một `ENG.txt`, script dùng file đó. Nếu không, dùng
`$PERSIST_ROOT/secrets/ENG.txt`, hoặc export `ENG_FILE=/path/to/ENG.txt`.
`ENG_URL` là lựa chọn thay thế: phải là URL HTTPS trực tiếp trả JSON mapping,
không phải trang Google Drive/HTML hoặc ZIP. Chỉ chọn `ENG_FILE` hoặc `ENG_URL`.
Không có URL ENG mặc định chưa được kiểm chứng. Lấy mapping từ
[AGQA supporting data](https://drive.google.com/drive/folders/1OMqA90VXY3BQorKFK5xWLSEEkqX31ui-),
được dẫn trên [trang AGQA](https://cs.stanford.edu/people/ranjaykrishna/agqa/).

Raw scene graphs là pickle: chỉ dùng nguồn dữ liệu tin cậy. SHA-256 được lưu ở
fresh run để phát hiện thay đổi khi resume, không chứng minh tính xác thực của
nguồn tải đầu tiên.

## 3. Submit fresh run

```bash
cd "$PERSIST_ROOT/code/DyGEnc"
mkdir -p logs
export TARGET_EPOCHS=1
export RUN_DIR="$PERSIST_ROOT/runs/dygenc/agqa_full"
sbatch scripts/slurm/train_agqa_full_ram.slurm
```

Mặc định: partition `batch`, 1 GPU, 8 CPU, 96 GiB RAM, 2 ngày. Nếu cần nhiều RAM
hơn và trường cho phép, override lúc submit:

```bash
sbatch --mem=160G scripts/slurm/train_agqa_full_ram.slurm
```

Đây là cấu hình tài nguyên ví dụ, **không đảm bảo full AGQA vừa 96/160 GiB**.
Không cần Conda/module: `/usr/bin/python3` 3.10–3.12 tạo venv trong RAM. Nếu thiếu
stdlib venv, script tải `virtualenv.pyz` vào RAM. PyTorch 2.5.0/cu121 và các gói
server được cài mới mỗi job. Compute node cần outbound Internet tới PyPI,
PyTorch/PyG, Kaggle và Hugging Face khi setup.

Luồng của một job:

1. Khóa `RUN_DIR`; kiểm tra tmpfs, native BF16 GPU, memory headroom và credentials.
2. Tạo môi trường/cache trong private tmpfs; tải hai model bằng immutable commit.
3. Tải AGQA vào tmpfs, chuẩn hóa file và kiểm tra hash.
4. Preprocess ModernBERT; không lưu bản NetworkX dư thừa.
5. Train; graph/description đọc từ tmpfs qua cache nhỏ hai video, không giữ thêm
   một bản toàn bộ graph tensors trong Python heap.
6. Lưu checkpoint ra disk; khi thoát, dọn đúng RAM directory của job này.

## 4. Theo dõi và resume

```bash
squeue -u "$USER"
tail -f logs/dygenc_agqa_full-<JOB_ID>.out
tail -f logs/dygenc_agqa_full-<JOB_ID>.err
sacct -j <JOB_ID> --format=JobID,State,ExitCode,Elapsed,MaxRSS
```

Checkpoint ở:

```text
$RUN_DIR/last.pth
$RUN_DIR/best.pth
```

Khi hết giờ/ngắt job, giữ nguyên `RUN_DIR`, `TARGET_EPOCHS`, source, model/data và
các thiết lập training rồi submit:

```bash
sbatch scripts/slurm/resume_agqa_full_ram.slurm
```

`TARGET_EPOCHS` là **tổng epoch đã đặt từ fresh run**, không phải số epoch chạy
thêm; không đổi giá trị này khi resume vì nó quyết định lịch learning rate.
Runner kiểm tra config/source, phiên bản runtime/GPU, hash raw data, seed và model
revision. Các path tmpfs thay đổi giữa hai job được phép. Không migrate checkpoint
GRN, hoặc `.pth` tạo bởi `train.py` cũ, sang định dạng này.

Resume khôi phục trainable parameters, tất cả model buffers, optimizer, RNG,
thứ tự dữ liệu/cursor và cả tiến độ validation. Trọng số base LLM bị freeze không
được ghi vào checkpoint; mỗi job tải lại đúng immutable revision của run gốc.
Mặc định batch=1, workers=0, accumulation=1. Không dùng nhánh gradient accumulation
có lỗi trong `train.py` cũ. Không có early stopping trong runner mới.

Slurm gửi `USR1` trước hết giờ 180 giây; runner cố hoàn tất optimizer update hiện
tại rồi lưu `last.pth`. Có thể yêu cầu dừng có checkpoint bằng:

```bash
scancel --batch --signal=USR1 <JOB_ID>
```

Exit khác 0 khi được yêu cầu dừng là chủ ý, không tự requeue. Nếu OOM/SIGKILL hoặc
một update kéo dài quá thời gian còn lại, chỉ checkpoint hoàn chỉnh gần nhất
được bảo toàn: tối đa mất tiến độ từ checkpoint đó. Nếu bị ngắt trong setup/
preprocess của fresh run, chưa có checkpoint; submit fresh lại. `best.pth` chỉ
xuất hiện sau validation hoàn chỉnh.

CUDA scatter/reduction và việc tạo lại embedding có thể không bitwise giống nhau;
resume đảm bảo khôi phục trạng thái/tiến độ, không hứa bitwise reproducibility GPU.

## 5. RAM/disk/GPU: giới hạn quan trọng

- `/dev/shm` là tmpfs nhưng có giới hạn dung lượng riêng; xem `df -h /dev/shm`.
  Toàn bộ tmpfs + Python heap + venv + model cache cùng tính vào giới hạn RAM/cgroup
  của job. Archive + dữ liệu giải nén có thể cùng tồn tại lúc tải.
- `RAM_BASE` phải là **tmpfs có quyền exec**. Nếu `/dev/shm` mount `noexec`, Python
  extension trong venv không load được: script dừng, không chuyển sang disk.
  Nhờ admin cung cấp executable tmpfs rồi export `RAM_BASE=/that/tmpfs`.
- tmpfs có thể bị swap theo cấu hình hệ điều hành. Nếu yêu cầu không ghi xuống
  thiết bị lưu trữ kể cả swap, cần admin cấu hình tmpfs/job không swap. Script
  không tự chạy `sudo`, mount hay tắt swap trên server dùng chung.
- Giữ đủ disk cho `last.pth` + `best.pth` + **một checkpoint tạm** khi thay file
  nguyên tử (peak khoảng 3 lần kích thước checkpoint). File tạm do lần SIGKILL
  trước được dọn có kiểm tra khi runner vào run dưới lock. Không sinh checkpoint
  cho từng epoch/step. Slurm log và pip-freeze/manifest nhỏ vẫn cần chỗ trống.
- Cleanup chạy khi thoát bình thường hoặc signal được xử lý. SIGKILL/node crash
  không cho cleanup chạy; RAM workspace có thể còn lại đến khi admin/node dọn.
  Chỉ dọn thủ công **đúng đường dẫn RAM workspace được in trong log của job đã
  kết thúc**, không xóa toàn bộ `/dev/shm` hay workspace của người khác.
- GPU cần native BF16 và NVIDIA driver tương thích CUDA 12.1 wheels. CPU RAM lớn
  không chữa CUDA OOM. T4/P100 bị từ chối; bản gốc hardcode một GPU và không có
  quantization/CPU fallback. Con số `max_memory=80GiB` trong model không tạo thêm VRAM.
- Cả train/test QA JSON vẫn được đọc vào RAM. Lazy graph cache giảm bản sao tensor,
  không loại bỏ chi phí đó. Nếu OOM, tăng RAM/VRAM hoặc cần tối ưu thêm pipeline;
  giảm `MIN_*` chỉ bỏ kiểm tra, không làm chương trình dùng ít RAM hơn.
- Theo upstream, `best.pth` chọn bằng loss trên official **test split**; không dùng
  nó để tuyên bố model selection trên held-out test độc lập.

## Biến cấu hình chính

| Biến | Mặc định |
| --- | --- |
| `PERSIST_ROOT` | `/media02/lnthanh03` |
| `SOURCE_REPO` | checkout tại thư mục submit |
| `RUN_DIR` | `$PERSIST_ROOT/runs/dygenc/agqa_full` |
| `RAM_BASE` | `/dev/shm` (bắt buộc executable tmpfs) |
| `PYTHON_BIN` | `/usr/bin/python3` |
| `KAGGLE_DATASET` | `tdat1465/agqa-balanced` |
| `KAGGLE_JSON`, `HF_TOKEN_FILE` | `$PERSIST_ROOT/secrets/kaggle.json`, `.../hf_token` |
| `ENG_FILE`, `ENG_URL` | mapping từ dataset, hoặc fallback nhỏ do bạn cung cấp |
| `TARGET_EPOCHS` | `1`, không đổi giữa fresh/resume |
| `CHECKPOINT_EVERY` | `1000` optimizer updates (validation: samples) |
| `SEED` | `18` |
| `DYGENC_EMBED_BATCH_SIZE` | `64` (không đổi khi resume) |
| `DYGENC_GRAPH_CACHE_SIZE` | `2` video |
| `MIN_RAM_FREE_GB` | `40` GiB headroom ban đầu |
| `MIN_TRAIN_HEADROOM_GB` | `24` GiB trước preprocess/train |
| `LLM_REVISION`, `MBERT_REVISION` | resolve `main` một lần khi fresh; resume dùng SHA đã lưu |

## Kiểm thử cục bộ

```bash
bash -n scripts/slurm/agqa_full_job.sh
bash -n scripts/slurm/train_agqa_full_ram.slurm
bash -n scripts/slurm/resume_agqa_full_ram.slurm
python -m unittest discover -s tests -v
```

Test staging dùng stdlib và dữ liệu giả; test trainer cần PyTorch/NumPy; test
dataset cần thêm loguru/tqdm. Không tải Llama/AGQA hay cần GPU để chạy unit tests.
Các test CPU không thay thế một lần kiểm tra full pipeline trên Slurm/GPU thực tế.
