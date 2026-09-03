# DyGEnc trên gpu03: A100, BF16, xin dưới 100 GB RAM

Bản `a100-90g-v1` dành cho tài khoản `nnthao21`, partition `batch`, node
`gpu03`, Python hệ thống 3.10.12 và một GPU có native BF16. Mỗi job xin
**`--mem=90G`**: 90 GiB, khoảng **96,6 GB thập phân**. Đây là mức tài nguyên
yêu cầu Slurm cấp, **chưa phải số đo chứng minh toàn bộ AGQA chạy vừa**.
Không chạy trực tiếp ngoài allocation và không tự đặt `CUDA_VISIBLE_DEVICES=0`.

## Cách giảm bộ nhớ mà không chuyển sang FP16/quantization

- Chỉ tải `AGQA_balanced.zip` và `AGQA_scene_graphs.zip` từ
  [dataset Hugging Face của bạn](https://huggingface.co/datasets/tdat1465/agqa-balanced).
  Không tải video Charades vì pipeline scene graph này không đọc video.
  Giải nén lần lượt và bỏ ZIP đã dùng. Kích thước file nén/raw không phải
  peak RAM: Python object, model và dữ liệu preprocess còn cần bộ nhớ riêng.
- Đọc QA JSON theo luồng, tạo chỉ mục SQLite **trong tmpfs**; không giữ toàn bộ
  train/test QA thành hai Python dictionary lớn. Giữ nguyên QA, thứ tự và grounding.
- Preprocess scene graph từng split, giải phóng split trước khi chuyển split sau;
  cache graph tensor nhỏ, mặc định hai video. Pickle của một split vẫn phải nạp
  nguyên khối, nên vẫn có peak RAM cần đo trên server.
- Bật gradient checkpointing chỉ trong Llama, giữ RNG/dropout khi tính lại.
  Không checkpoint GNN có BatchNorm. Giữ chính sách dtype gốc: BF16 autocast,
  không ép các tham số FP32 của PEFT/GNN xuống precision thấp hơn.
- Microbatch 1, cộng dồn gradient 32 mẫu; backward từng mẫu rồi chuẩn hóa theo
  tổng token được giám sát trước gradient clipping. Không giữ 32 đồ thị GPU.

Venv, dependency, data, preprocessing, model tải về và cache nằm trong private
tmpfs của **cùng một job**. Disk chỉ giữ code, checkpoint, log/metadata nhỏ và
HF token. Không có fallback tải data/model xuống disk. CPU RAM không thay GPU VRAM.

## 1. Clone đúng phiên bản vào thư mục của nnthao21

Chạy trên server, không phải trong PowerShell máy cá nhân:

```bash
umask 077
export PERSIST_ROOT=/media02/lnthanh03/nnthao21
export SOURCE_REPO="$PERSIST_ROOT/code/DyGEnc-a100-90g-v1"
mkdir -p "$PERSIST_ROOT/code" "$PERSIST_ROOT/secrets"
chmod 700 "$PERSIST_ROOT/secrets"

GIT_TERMINAL_PROMPT=0 git clone --depth 1 --branch a100-90g-v1 \
  https://github.com/tdat1465/DyGEnc.git "$SOURCE_REPO"
cd "$SOURCE_REPO"
mkdir -p logs
```

Repo công khai nên clone không cần GitHub username/token. Nếu thư mục clone đã
tồn tại, dùng lại đúng checkout hoặc chọn tên thư mục mới, không xóa run cũ.
Thư mục run và file token phải **thuộc UID của nnthao21**, không chỉ có quyền ghi
vào thư mục của `lnthanh03`. Không đổi ownership của thư mục dùng chung.

## 2. Chuẩn bị một HF read token nhỏ

Tài khoản Hugging Face cần được cấp quyền tải
[meta-llama/Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B).
Lưu token một dòng bằng editor hoặc upload file riêng; không đưa token vào Git,
log, notebook công khai hay tham số `sbatch`.

```bash
nano "$PERSIST_ROOT/secrets/hf_token"
chmod 600 "$PERSIST_ROOT/secrets/hf_token"
```

**Không cần `kaggle.json`.** Hai archive AGQA được lấy trực tiếp từ Hugging Face,
pin immutable revision ở fresh run và kiểm tra SHA-256 archive.

Script tự tải `ENG.txt` bằng URL đã kiểm tra trả JSON mapping từ
[AGQA supporting data](https://drive.google.com/drive/folders/1OMqA90VXY3BQorKFK5xWLSEEkqX31ui-).
Nếu Google Drive bị chặn hoặc bạn đã có file đúng, cung cấp file nhỏ này:

```bash
export ENG_FILE="$PERSIST_ROOT/secrets/ENG.txt"
unset ENG_URL
```

Chỉ đặt `ENG_FILE` khi file thực sự tồn tại. Mặc định script cũng tự nhận file tại
vị trí trên nếu có. Mapping phải là JSON, không phải HTML tải nhầm. Manifest raw
ghi hash mapping và bốn input để resume phát hiện thay đổi. Scene graph là pickle:
chỉ dùng dataset có nguồn tin cậy; hash không chứng minh tính an toàn của pickle.

## 3. Chọn profile và chạy thử hai optimizer updates

Hai wrapper A100 mặc định dùng **`TRAIN_PROFILE=upstream`**: giữ ngưỡng sequence
trong config gốc (train 50, test 10). Đây **không phải full mọi QA**.
Chế độ `full` giữ mọi QA có grounding không rỗng; xem phần riêng bên dưới.

```bash
cd "$SOURCE_REPO"
export RUN_DIR="$PERSIST_ROOT/runs/dygenc/a100_upstream_90g"
export TRAIN_PROFILE=upstream
export TARGET_EPOCHS=5
export ACCUMULATION_STEPS=32
export LOSS_REDUCTION=token_mean
export STOP_AFTER_UPDATES=2
export CHECKPOINT_EVERY=1

# Chỉ kiểm tra yêu cầu tài nguyên, chưa tải/cài/train:
sbatch --test-only scripts/slurm/train_agqa_a100_90g.slurm

# Nếu Slurm chấp nhận, submit job thật:
sbatch scripts/slurm/train_agqa_a100_90g.slurm
```

Mặc định: partition `batch`, node `gpu03`, **1 GPU, 8 CPU, 90G RAM**, tối đa 2 ngày.
`--test-only` không kiểm tra dataset/token/peak RAM. Trạng thái node `mix` nghĩa là
đã có tài nguyên đang dùng; `sinfo` ghi `gpu:4` không bảo đảm bốn A100 đang trống.
Job sẽ kiểm tra đúng một GPU được nhìn thấy và có native BF16 trước khi tải model.

Chạy thử vẫn **tải và preprocess đầy đủ dữ liệu cần cho profile**, không lấy một
subset QA để giả lập vừa RAM. Sau hai optimizer updates (thường 64 mẫu), runner
lưu `last.pth` rồi thoát **75 có chủ ý**. Slurm có thể hiển thị `FAILED`/`75:0`;
đọc log để phân biệt việc dừng chủ động với lỗi. Đây là kiểm tra pipeline bước đầu,
không bảo đảm các QA dài hơn hoặc validation sau này sẽ không OOM.

Không cần Conda/module: script dùng `/usr/bin/python3`, tạo môi trường trong RAM,
cài PyTorch 2.5.0/cu121 cùng dependency pin. Nếu thiếu stdlib venv, có fallback
`virtualenv.pyz` cũng trong RAM. Compute node cần Internet tới PyPI,
PyTorch/PyG, Hugging Face và Google Drive (nếu tải ENG tự động).

## 4. Xem log, rồi resume để train đủ 5 epoch

Thay `12345` bằng job ID thực tế:

```bash
squeue -u "$USER"
tail -n 80 -f logs/dygenc_a100_90g-12345.out logs/dygenc_a100_90g-12345.err
```

Nhấn `Ctrl+C` để thoát **tail**, không hủy job. Sau đó xem trạng thái kết thúc:

```bash
sacct -j 12345 --format=JobID,State,ExitCode,Elapsed,MaxRSS
ls -lh "$RUN_DIR/last.pth"
```

Checkpoint được ghi ở `$RUN_DIR/last.pth`. Log checkpoint có peak CUDA allocated/
reserved; `MaxRSS` tùy cấu hình accounting và không nhất thiết phản ánh đủ mọi
trang tmpfs dùng chung. Kiểm tra thêm log OOM/cgroup nếu job bị kill.

Khi chạy thử đã kết thúc thành công theo log và có `last.pth`, giữ nguyên source,
`RUN_DIR`, profile, epoch, accumulation, reduction và seed, rồi chạy:

```bash
export STOP_AFTER_UPDATES=0
export CHECKPOINT_EVERY=1000
sbatch scripts/slurm/resume_agqa_a100_90g.slurm
```

Dùng đúng **`resume_agqa_a100_90g.slurm`** cho cặp A100; không đổi sang wrapper
generic có mặc định profile/run path khác. Không submit resume khi job cũ còn chạy.
`TARGET_EPOCHS=5` là tổng số epoch của run, không phải 5 epoch bổ sung.

Nếu mở **SSH mới**, dùng nguyên khối dưới đây để tiếp tục run `upstream` đã tạo
ở bước 3; không clone lại hoặc submit fresh vào cùng thư mục:

```bash
export PERSIST_ROOT=/media02/lnthanh03/nnthao21
export SOURCE_REPO="$PERSIST_ROOT/code/DyGEnc-a100-90g-v1"
export RUN_DIR="$PERSIST_ROOT/runs/dygenc/a100_upstream_90g"
export TRAIN_PROFILE=upstream
export TARGET_EPOCHS=5
export ACCUMULATION_STEPS=32
export LOSS_REDUCTION=token_mean
export STOP_AFTER_UPDATES=0
export CHECKPOINT_EVERY=1000
cd "$SOURCE_REPO"
test -s "$RUN_DIR/last.pth" && sbatch scripts/slurm/resume_agqa_a100_90g.slurm
```

Log job resume có tên `logs/dygenc_a100_resume-<job-id>.out` và `.err`.
Nếu trước đó dùng `ENG_FILE`, `HF_TOKEN_FILE`, seed hoặc model revision tùy chỉnh,
khai báo lại đúng các giá trị đó. Không sửa code/dependency giữa một run và resume.

Mỗi job resume dựng lại môi trường, tải đúng revision data/model và preprocess
trong RAM; đây là đánh đổi để không lưu chúng trên disk. Trainer khôi phục
trainable parameters, model buffers, optimizer, RNG, thứ tự/cursor dữ liệu và
tiến độ validation. Không ghi base LLM đã freeze vào checkpoint. GPU/reduction
có thể không bitwise deterministic, dù trạng thái được khôi phục.

Để yêu cầu dừng có checkpoint:

```bash
scancel --batch --signal=USR1 12345
```

Script cũng xin Slurm gửi USR1 trước hết giờ 180 giây. Runner cố hoàn tất update
hiện tại rồi lưu; SIGKILL/OOM hoặc hết giờ quá nhanh có thể chỉ giữ được checkpoint
hoàn chỉnh trước đó. Không tự requeue. Nếu thất bại trước checkpoint đầu tiên,
chọn **RUN_DIR mới** để submit fresh; giữ log/metadata cũ để chẩn đoán.

## Nếu muốn full AGQA, không dùng ngưỡng 50/10

Đặt rõ profile và thư mục run mới, rồi vẫn dùng cặp wrapper A100 90G:

```bash
cd "$SOURCE_REPO"
export TRAIN_PROFILE=full
export RUN_DIR="$PERSIST_ROOT/runs/dygenc/a100_full_90g"
export TARGET_EPOCHS=5
export ACCUMULATION_STEPS=32
export LOSS_REDUCTION=token_mean
export STOP_AFTER_UPDATES=2
export CHECKPOINT_EVERY=1
sbatch scripts/slurm/train_agqa_a100_90g.slurm
```

**Đợi probe kết thúc**, kiểm tra log và `last.pth` trước khi chạy khối tiếp theo:

```bash
export STOP_AFTER_UPDATES=0
export CHECKPOINT_EVERY=1000
test -s "$RUN_DIR/last.pth" && sbatch scripts/slurm/resume_agqa_a100_90g.slurm
```

Không đổi profile giữa một run và lần resume. Không tự lọc bớt QA khi thiếu RAM.
Nếu mở SSH mới để resume full, dùng khối khai báo lại ở bước 4 nhưng sửa
`TRAIN_PROFILE=full` và `RUN_DIR="$PERSIST_ROOT/runs/dygenc/a100_full_90g"`.
Các wrapper cũ `train_agqa_full_ram.slurm` / `resume_agqa_full_ram.slurm` vẫn có
profile mặc định `full` và nay cũng xin 90G, nhưng `PERSIST_ROOT` cũ là
`/media02/lnthanh03`; nên dùng cặp A100 ở trên để có mặc định subtree của nnthao21.

## Độ chính xác và giới hạn tái lập paper

Các thay đổi lưu trữ/checkpointing không chủ đích đổi dữ liệu hay hạ precision.
Nhưng **chưa thể cam kết accuracy bằng paper**, kể cả khi giữ BF16:

- [Paper](https://arxiv.org/html/2505.03581v1#S4.SS3) mô tả batch 32, 5 epoch,
  early stopping patience 2, A100 80GB. Bản này đặt 5 epoch/accumulation 32 trên
  A100 40GB; microbatch 1 không tương đương tuyệt đối physical batch 32 khi mô hình
  có BatchNorm. Không thay đổi kiến trúc GNN để che giấu khác biệt đó.
- Paper mô tả ModernBERT-base, trong khi implementation upstream dùng
  ModernBERT-large và feature dimensions tương ứng. Bản này giữ **implementation
  hiện tại**, không tự thay model hoặc kích thước feature.
- Runner dùng loss trung bình theo token cho accumulation; không tái sử dụng
  monkey-patch loss có lỗi của `train.py` cũ. Đây là khác biệt training có chủ ý.
- Runner không early stopping. Theo config upstream, validation và lựa chọn
  `best.pth` dùng official **test split**; không dùng điều đó để tuyên bố model
  selection trên held-out test độc lập. Muốn so sánh paper cần thống nhất thêm
  protocol train/validation/evaluation và đo accuracy thực tế.

Không resume checkpoint `kaggle-ram-v1`, `train.py` cũ hoặc GRN bằng phiên bản này.
Source/runtime/loss contract đã đổi; runner chủ động từ chối resume không tương
thích. Giữ checkout cũ nếu cần hoàn thành run cũ.

## Những giới hạn cần biết

- 90G bao gồm heap Python, tmpfs data/model/cache, preprocessing và môi trường,
  không chỉ mỗi dữ liệu. Slurm cần cấu hình cgroup enforcement để thực thi giới
  hạn; `--mem` tự nó không phải bộ đo peak hoặc cơ chế giới hạn của Python.
- `/dev/shm` 126G là dung lượng mount dùng chung, **không phải** 126G dành riêng
  cho job. `RAM_BASE` phải là tmpfs có quyền exec; nếu noexec, script dừng.
- tmpfs có thể bị **swap xuống disk** theo cấu hình OS. Script không tự tắt swap
  trên server dùng chung. Nếu cần tuyệt đối không ghi thiết bị lưu trữ kể cả swap,
  phải nhờ admin cấu hình cho job/mount.
- Giữ disk cho `last.pth`, `best.pth` và một checkpoint tạm lúc thay nguyên tử
  (peak khoảng ba lần kích thước checkpoint), cộng log/metadata nhỏ.
- SIGKILL/node crash có thể không chạy cleanup. Chỉ dọn đúng private RAM path
  được in trong log của job đã kết thúc, không xóa toàn bộ `/dev/shm`.
- Nếu OOM ở 90G, gửi phần log lỗi và `sacct` để xác định heap/tmpfs hay GPU.
  Không giảm ngưỡng kiểm tra RAM hoặc chuyển FP16 để coi như đã xử lý vấn đề.

## Các biến bổ sung

| Biến | Mặc định / ý nghĩa |
| --- | --- |
| `RAM_BASE` | `/dev/shm`, executable tmpfs |
| `PYTHON_BIN` | `/usr/bin/python3`, hỗ trợ 3.10–3.12 |
| `HF_TOKEN_FILE` | `$PERSIST_ROOT/secrets/hf_token`, đúng UID và mode 600 |
| `DATASET_REPO` | `tdat1465/agqa-balanced` trên Hugging Face |
| `DATASET_REVISION` | resolve `main` một lần; resume dùng SHA đã lưu |
| `LLM_REVISION`, `MBERT_REVISION` | resolve một lần; resume dùng SHA đã lưu |
| `SEED` | `18` |
| `DYGENC_EMBED_BATCH_SIZE` | `64`, không đổi giữa fresh/resume |
| `DYGENC_GRAPH_CACHE_SIZE` | `2` video |
| `MIN_RAM_FREE_GB` | `40` GiB headroom ban đầu, không phải giới hạn dùng RAM |
| `MIN_TRAIN_HEADROOM_GB` | `24` GiB headroom trước preprocess/train |

File chính: `agqa_full_job.sh` quản lý allocation/RAM/credentials;
`download_agqa.py` tải chọn lọc và kiểm tra archive; `stage_agqa.py` chuẩn hóa năm
input; `src/server_train.py` train/resume. Notebook Kaggle cũ vẫn pin
`kaggle-ram-v1`, không phải hướng dẫn chạy A100 90G này.

## Kiểm thử cục bộ

```bash
bash -n scripts/slurm/agqa_full_job.sh
bash -n scripts/slurm/train_agqa_a100_90g.slurm
bash -n scripts/slurm/resume_agqa_a100_90g.slurm
python -m unittest discover -s tests -v
```

Test dùng dữ liệu giả, CPU và tiny Llama/LoRA khởi tạo ngẫu nhiên (khi đủ dependency),
không tải Llama/AGQA. Chúng kiểm tra thứ tự/grounding indexed QA, tải ZIP an toàn,
gradient/dropout khi checkpoint, token accumulation và resume. **Chưa thay thế
chạy thực tế trên A100 với toàn bộ AGQA.**

Kiểm chứng bản này trên Windows/Python 3.12.13, PyTorch 2.5.0 CPU,
Transformers 4.50.1, PEFT 0.15.2: **95 tests passed, 2 skipped** (hai test cần
quyền tạo symlink không có trên máy kiểm thử). Tiny Llama/LoRA thực sự chạy,
không bị skip. `pip check` và kiểm tra cú pháp cả năm script Bash đều qua.
Chưa kiểm thử allocation Slurm, CUDA wheel hoặc peak RAM/accuracy trên gpu03.
