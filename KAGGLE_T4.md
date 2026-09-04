# Chạy smoke test DyGEnc trên Kaggle T4

Branch `codex/kaggle-t4` dành riêng để kiểm tra pipeline trên Kaggle. Cấu hình
mặc định chỉ preprocess **8 video và tối đa 128 QA cho mỗi split**, sau đó chạy
**2 optimizer updates** rồi lưu checkpoint. Đây không phải full-data training và
không được dùng kết quả của nó để so sánh accuracy với paper.

Notebook sẵn dùng: `notebooks/kaggle_t4.ipynb`.

## 1. Chuẩn bị Kaggle Notebook

Trong phần Settings của notebook:

1. Chọn accelerator có Tesla T4. Notebook đã kiểm tra với runtime mục tiêu
   Python 3.12 và PyTorch `2.10.0+cu128`.
2. Bật Internet.
3. Chọn **Add Input** và thêm dataset
   `tdat1465/agqa-balanced`.
4. Mở trang model `meta-llama/Llama-3.2-3B`, chấp nhận điều khoản và bảo đảm
   tài khoản Hugging Face của bạn đã được cấp quyền.
5. Tạo Kaggle Secret tên chính xác `HF_TOKEN`, chứa một Hugging Face **read
   token**, rồi bật quyền dùng secret cho notebook.

Không cần GitHub token vì repository và branch là công khai. Không dán
`HF_TOKEN` trực tiếp vào cell hay URL.

Dataset được mount tại:

```text
/kaggle/input/datasets/tdat1465/agqa-balanced
```

Launcher chỉ tìm bốn file trong hai cây `AGQA_balanced` và
`AGQA_scene_graphs`; nó không quét hoặc copy thư mục video Charades. Nếu
dataset thiếu `ENG.txt`, launcher tải bản chính thức và chỉ nhận file có
SHA-256 đã biết.

## 2. Clone đúng branch, không hiện hộp hỏi username

Chạy cell sau trong một Kaggle session mới:

```python
from pathlib import Path
import os
import subprocess

REPO_URL = "https://github.com/tdat1465/DyGEnc.git"
BRANCH = "codex/kaggle-t4"
REPO_DIR = Path("/kaggle/working/DyGEnc-kaggle-t4")

git_env = os.environ.copy()
git_env["GIT_TERMINAL_PROMPT"] = "0"

if not REPO_DIR.exists():
    subprocess.run([
        "git", "clone", "--depth", "1", "--single-branch",
        "--branch", BRANCH, REPO_URL, str(REPO_DIR),
    ], env=git_env, check=True)
else:
    assert (REPO_DIR / ".git").is_dir(), "REPO_DIR tồn tại nhưng không phải Git checkout"
    assert subprocess.run(
        ["git", "-C", str(REPO_DIR), "status", "--porcelain"],
        env=git_env, check=True, text=True, capture_output=True,
    ).stdout == "", "Repo có thay đổi chưa lưu"
    subprocess.run([
        "git", "-C", str(REPO_DIR), "fetch", "--depth", "1",
        "origin", f"refs/heads/{BRANCH}",
    ], env=git_env, check=True)
    subprocess.run([
        "git", "-C", str(REPO_DIR), "checkout", "--detach", "FETCH_HEAD",
    ], env=git_env, check=True)

commit = subprocess.run(
    ["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"],
    env=git_env, check=True, text=True, capture_output=True,
).stdout.strip()
print("Code:", REPO_DIR)
print("Branch source commit:", commit)
```

`GIT_TERMINAL_PROMPT=0` biến lỗi clone thành lỗi rõ ràng thay vì treo ở dòng
`Username for 'https://github.com'`.

## 3. Chạy probe hai optimizer updates

```python
from kaggle_secrets import UserSecretsClient

MOUNTED_ROOT = Path("/kaggle/input/datasets/tdat1465/agqa-balanced")
WORK_ROOT = Path("/kaggle/working/dygenc-t4-work")
RUN_DIR = Path("/kaggle/working/dygenc-t4-run")

command = [
    "python", str(REPO_DIR / "scripts/kaggle/run_agqa_t4.py"),
    "--mounted-root", str(MOUNTED_ROOT),
    "--work-root", str(WORK_ROOT),
    "--run-dir", str(RUN_DIR),
    "--gpu-index", "0",
    "--mode", "fresh",
    "--smoke-videos-per-split", "8",
    "--smoke-qa-per-split", "128",
    "--accumulation-steps", "32",
    "--stop-after-updates", "2",
    "--checkpoint-every", "1",
]

child_env = os.environ.copy()
token = UserSecretsClient().get_secret("HF_TOKEN")
child_env["HF_TOKEN"] = token
process = subprocess.Popen(command, env=child_env)
try:
    returncode = process.wait()
except KeyboardInterrupt:
    # Launcher chuyển SIGTERM cho trainer và đợi checkpoint an toàn.
    process.terminate()
    returncode = process.wait()
finally:
    child_env.pop("HF_TOKEN", None)
    del token, child_env

if returncode not in (0, 75):
    raise RuntimeError(f"Launcher failed with exit code {returncode}")
if returncode == 75:
    print("Probe hoàn tất đúng dự kiến: checkpoint đã lưu trước exit 75.")
assert (RUN_DIR / "last.pth").is_file()
print("Checkpoint:", RUN_DIR / "last.pth")
```

Lần đầu có thể lâu vì launcher phải:

1. tạo venv bằng `--without-pip`, rồi dùng lại Torch CUDA và pip hệ thống của
   Kaggle (không gọi bootstrap `ensurepip`);
2. cài đúng wheel `torch_scatter==2.1.2+pt210cu128` từ PyG;
3. hash bốn input lớn và tải hai model ở revision bất biến;
4. unpickle từng scene-graph split nhưng chỉ encode/lưu 8 video đã chọn;
5. chạy 64 microbatch cho 2 update khi accumulation là 32.

Dòng kết thúc mong đợi:

```text
Requested optimizer-update probe completed; resume checkpoint saved (exit 75).
```

Exit `75` là điểm dừng có chủ đích, không phải OOM. Traceback, `CUDA out of
memory`, hoặc không có `last.pth` mới là thất bại.

## 4. Resume trong cùng Kaggle session

Giữ nguyên dataset, branch, run directory, số video/QA, accumulation và epochs.
Chỉ đổi mode và số update dừng:

```python
command = [
    "python", str(REPO_DIR / "scripts/kaggle/run_agqa_t4.py"),
    "--mounted-root", str(MOUNTED_ROOT),
    "--work-root", str(WORK_ROOT),
    "--run-dir", str(RUN_DIR),
    "--gpu-index", "0",
    "--mode", "resume",
    "--smoke-videos-per-split", "8",
    "--smoke-qa-per-split", "128",
    "--accumulation-steps", "32",
    "--stop-after-updates", "0",
    "--checkpoint-every", "1",
]

child_env = os.environ.copy()
token = UserSecretsClient().get_secret("HF_TOKEN")
child_env["HF_TOKEN"] = token
process = subprocess.Popen(command, env=child_env)
try:
    returncode = process.wait()
except KeyboardInterrupt:
    process.terminate()
    returncode = process.wait()
finally:
    child_env.pop("HF_TOKEN", None)
    del token, child_env

if returncode not in (0, 75):
    raise RuntimeError(f"Resume failed with exit code {returncode}")
print("Resume exit:", returncode)
```

`stop-after-updates=0` chạy hết tập smoke và validation qua 5 epoch. Nó vẫn
không chuyển thành full-data run.

## 5. Resume ở một Kaggle session khác

Trước khi session cũ hết hạn, dùng **Save Version** để lưu output. Ở notebook
mới, Add Input output đó và copy các file nhỏ thuộc chính run của bạn vào
`RUN_DIR`:

```python
import shutil

# Sửa thành đường dẫn thật hiển thị trong Data Explorer.
SAVED_RUN_DIR = Path("/kaggle/input/<your-saved-output>/dygenc-t4-run")
RUN_DIR.mkdir(parents=True, exist_ok=True)

for name in (
    "last.pth", "best.pth", "model-revisions.json",
    "raw-data-manifest.json", "pip-freeze.txt",
):
    source = SAVED_RUN_DIR / name
    if source.is_file():
        shutil.copy2(source, RUN_DIR / name)

assert (RUN_DIR / "last.pth").is_file()
assert (RUN_DIR / "model-revisions.json").is_file()
```

Sau đó chạy cell resume ở mục 4. Preprocessing subset và model cache sẽ được
tạo lại; checkpoint tiếp tục đúng cursor/optimizer/scaler đã lưu.

Chỉ load `last.pth` do chính bạn tạo. Checkpoint PyTorch và hai file scene graph
AGQA đều có cấu trúc pickle, nên dataset/checkpoint phải đến từ nguồn bạn tin
cậy; không thay input bằng file `.pkl` ngẫu nhiên.

## Cấu hình số học và giới hạn

- T4 không có native BF16, nên branch này dùng Llama **FP16 không quantize**,
  LoRA/trainable modules FP32 và AMP gradient scaling.
- Gradient checkpointing vẫn bật. Target-only causal loss tránh tạo logits cho
  toàn bộ token prompt; cross entropy đáp án vẫn tính FP32 và giữ cùng mục tiêu
  toán học.
- Chỉ GPU được chọn bởi `--gpu-index` hiện với tiến trình train. GPU T4 thứ hai
  không dùng; không có DDP hay thay đổi effective batch.
- Đây là đường kiểm tra tương thích/chức năng. Dùng branch/tag A100 BF16 riêng
  để chạy thí nghiệm cần so sánh với paper.
- Checkpoint T4 không tương thích với checkpoint A100 vì runtime, dtype và source
  contract khác nhau.

## Lỗi thường gặp

- `Missing HF_TOKEN`: chưa tạo/bật quyền Kaggle Secret hoặc sai tên secret.
- Lỗi có dòng `ensurepip --upgrade --default-pip`: notebook đang dùng commit cũ;
  fetch lại branch rồi chạy lại cell. Bản hiện tại dùng thư mục
  `venv-no-ensurepip-v1` nên không cần xóa `RUN_DIR` hay checkpoint.
- `401/403` khi tải Llama: tài khoản chứa token chưa được Meta cấp quyền model.
- `config.json is not a valid JSON file`: metadata trong Hugging Face cache bị
  hỏng sau một lần tải dở hoặc proxy trả sai `Content-Length: 0` cho HEAD. Fetch
  branch mới nhất rồi chạy lại. Khi Hub báo nhầm kích thước bằng 0,
  launcher sẽ GET trực tiếp file tại đúng revision, xác minh JSON rồi thay thế
  nguyên tử đúng blob lỗi; các weight đã tải thành công được giữ nguyên.
  Nếu lỗi vẫn còn, gửi nguyên dòng `Direct HTTPS metadata repair...` hoặc `Hub
  metadata remains invalid after atomic repair` (thông tin chẩn đoán không chứa
  token) để kiểm tra tiếp.
- Sai Torch/Python: Kaggle đã đổi image. Không ép cài wheel cũ; cần cập nhật
  branch và PyG wheel đồng bộ trước.
- `Expected exactly one ... below ...`: dataset mount thiếu file hoặc chứa bản
  trùng; kiểm tra Data Explorer.
- `Another Kaggle launcher is already using this run directory`: một cell khác
  vẫn đang chạy với cùng `RUN_DIR`; dừng hoặc chờ cell đó, không chạy song song.
- `Both training and validation need at least one eligible sample`: 8 video đầu
  không tạo đủ mẫu theo ngưỡng upstream. Dùng một `RUN_DIR` mới và tăng
  `--smoke-videos-per-split`, ví dụ 32.
- `CUDA out of memory`: accumulation thấp hơn không làm một sample ngắn lại.
  Giữ profile upstream, đóng kernel/process GPU khác và gửi phần log có dòng
  `GPU peak allocated/reserved` để điều chỉnh branch.
- Kaggle dừng cưỡng bức giữa một accumulation group: resume từ checkpoint hoàn
  chỉnh gần nhất; với `--checkpoint-every 1`, tối đa mất group đang chạy.
