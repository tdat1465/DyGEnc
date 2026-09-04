This repository extends the [upstream DyGEnc implementation](https://github.com/linukc/DyGEnc)
with RAM-only school-server jobs and resumable AGQA training. The original model
is DyGEnc (Dynamic Graph Encoding), a method for structured spatio-temporal
reasoning using sequences of textual scene graphs and large language models.

## Kaggle source setup

For a bounded compatibility test on a Kaggle Tesla T4, use branch
`codex/kaggle-t4` and run [the T4 notebook](notebooks/kaggle_t4.ipynb). The
complete setup, fresh-run, checkpoint, and resume instructions are in
[the Kaggle T4 guide](KAGGLE_T4.md). This path uses unquantized FP16 with
gradient scaling and is a smoke test, not a paper-accuracy result.

The older [clone-only notebook](notebooks/kaggle_clone.ipynb) targets the
`kaggle-ram-v1` source snapshot and does not provide T4 training support.

## RAM-only school-server jobs

See [the Slurm guide](scripts/slurm/README.md) for fresh/resume AGQA jobs that keep
the Python environment, downloaded data, preprocessing outputs and pretrained
model caches in executable tmpfs. Only checkpoints and small logs/metadata stay
on disk. These jobs use the separate resumable `src.server_train` runner, not the
legacy `train.py` below.

The `a100-90g-v1` server release requests **90 GiB RAM (about 96.6 decimal GB)**
and one native-BF16 GPU on `gpu03`. It downloads only the two required AGQA
archives from Hugging Face, streams QA into a RAM-backed SQLite index, and
checkpoints Llama activations without quantization. The A100 fresh/resume pair
defaults to upstream sequence filters, 5 epochs and accumulation 32; full QA is
an explicit profile. Run the resumable two-update probe before a long job.
Memory fit and paper-level accuracy still need validation on the real server.

## Installation

```bash
# Core dependencies
conda create -y -n dygenc python=3.10
conda activate dygenc
pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric
pip install transformers==4.50.1 sentencepiece
pip install peft
# Other dependencies
pip install -r requirements.txt
pip install -U "huggingface_hub[cli]"
# Optional dependencies
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cu121.html
pip install git+https://github.com/bfshi/scaling_on_scales.git
pip install --upgrade tokenizers==0.21.4
```

## Running from scratch

### Downloading

Follow instructions in the `datasets/` folder.

At the end call `source setup.bash` to set path to data or add argument for a custom path:

```bash
#!/bin/bash

PWD=${1:-`pwd`}

AGQA_ROOT=$PWD/agqa
STAR_ROOT=$PWD/star
```

### Data Preprocessing

```bash
# star
python -m src.datasets.preprocess.star
# agga
python -m src.datasets.preprocess.agga
```

### Training

Check `src/cfgs/<dataset_name>.py` before start:

```bash
python3 train.py --dataset_name=<(agqa|star)> --exp_name=<name> <optional args>
```

### Prediction

Check `src/cfgs/<dataset_name>.py` before start:

```bash
python3 predict.py --dataset_name=<(agqa|star)> --exp_name=<name> <optional args>
```

After this step, you can find results in the corresponding folder `eval/<dataset_name>/<exp_name>`.

### Evaluating

```bash
python3 eval.py --dataset_name=<(agqa|star)> --exp_name=<name>
```
