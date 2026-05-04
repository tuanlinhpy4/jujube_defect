# MvANet - Red Jujube Recognition

PyTorch implementation of Multi-vision Attention Network (MvANet) for red
jujube grading. The repo is organized around one reproducible CLI for train,
eval, inference, export, and model-complexity checks.

## Reproducibility Targets

This repo was tightened against common A*/Q1 artifact expectations:

- Paper setting is explicit in `configs/default.yaml`.
- Install is one command for core training: `pip install -r requirements.txt`.
- Train/eval/export commands are documented and share the same config path.
- Randomness is controlled for Python, NumPy, PyTorch, CUDA, and DataLoader
  workers.
- Each run saves `config.yaml`, `environment.json`, `dataset_summary.json`,
  checkpoints, metrics JSON, and plots.
- Checkpoints include optimizer, scheduler, config, and RNG state for resume.

See `REPRODUCIBILITY.md` for the submission checklist and exact seed protocol.

References used for the audit:

- Original paper/arXiv: https://arxiv.org/abs/1904.00388
- PyTorch reproducibility note: https://docs.pytorch.org/docs/2.11/notes/randomness.html
- NeurIPS paper checklist: https://neurips.cc/public/guides/PaperChecklist
- Papers with Code research-code checklist: https://github.com/paperswithcode/releasing-research-code
- ACM artifact badging criteria: https://www.acm.org/publications/policies/artifact-review-and-badging-current

## Project Layout

```text
configs/
  default.yaml          # paper-aligned single-run config
  smoke.yaml            # one-epoch smoke-test config
data/
  dataset.py            # transforms, seeded splits, DataLoaders
models/
  mvanet.py             # original architecture, all code in one file
  mvanet_opt.py         # optimized variant, all code in one file
  MvANet*/              # legacy import shims for older checkpoints/scripts
utils/
  reproducibility.py    # seeds, RNG state, environment metadata
  metrics.py            # accuracy, precision, recall, F1, confusion matrix
  logger.py             # epoch and multi-run logging
train.py                # main CLI: info/train/eval/export
eval.py                 # standalone eval wrapper
predict.py              # image or 5-image voting inference
export_onnx.py          # standalone ONNX export
count_flops.py          # optional ptflops complexity report
```

## Add A Model

New model architectures are single-file modules. Add one file under `models/`,
for example `models/my_model.py`, and export:

```python
ARCH_NAME = "MyModel"
MODEL_CONFIGS = {"small": {...}, "large": {...}}

def build_model(name: str, num_classes: int = 4, in_channels: int = 3, **overrides):
    ...
```

Then run:

```bash
python train.py train --arch MyModel --model small --data-root ./dataset
```

Do not create `blocks.py` plus `mvanet.py` for new models; keep the whole model
definition in the one architecture file.

## Setup

Use Python 3.9+.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Optional tools for ONNX export/verification and FLOPs:

```bash
pip install -r requirements-optional.txt
```

## Dataset

Expected class folder names and order:

```text
dataset/
  train/
    invalid/
    rotten/
    wizened/
    normal/
  test/
    invalid/
    rotten/
    wizened/
    normal/
```

If `dataset/val/` exists, it is used. If it does not exist, the code holds out
`data.val_size` images from `train/` with the run seed. The default value is
`1280`, matching the paper.

## Paper-Aligned Defaults

`configs/default.yaml` uses the reported MvANet paper setting where published:

- `MvANet / mvanet_3`
- input size `32x32`
- train batch size `64`
- eval batch size `5`
- epochs `300`
- optimizer `SGD`, momentum `0.9`
- initial learning rate `0.1`
- LR step decay by `0.1` at epochs `120`, `210`, `270`
- augmentation: horizontal mirroring and rotation

The paper does not publish dataset mean/std or weight decay. Defaults therefore
do not invent them: `normalize_mean: null`, `normalize_std: null`,
`weight_decay: 0.0`. If you use dataset-specific normalization, write the exact
values into the config before training.

## Commands

List available models:

```bash
python train.py info
```

Run the default paper-aligned single run:

```bash
python train.py train --data-root ./dataset
```

Train mode always evaluates the best checkpoint on `dataset/test/` after every
seed. A training dataset therefore needs both a validation split and a test
split. If `dataset/val/` is absent, `data.val_size` is held out from `train/`.

Run the fixed 5-seed protocol:

```bash
python train.py train --data-root ./dataset --seed-mode five
```

The five seeds are exactly `42, 123, 456, 789, 2024`. Single-seed mode uses
`42`. The aggregate `multirun_summary.json` reports mean +/- std for test error
and test accuracy across seeds. For any repeat count other than 1 or this fixed
5-seed protocol, use `--seed-mode range` or pass explicit `--seeds`.

Fast smoke test on a tiny local dataset:

```bash
python train.py train --config configs/smoke.yaml --data-root ./dataset
```

Evaluate/test a checkpoint. This mode is only for existing weights and requires
`--checkpoint`. For eval-only runs the data root may contain only `test/` or
`val/`; class order is read from the checkpoint and checked against the split
folders:

```bash
python train.py eval --checkpoint runs/mvanet_3/run_01/best.pth --data-root ./dataset
```

Predict one image:

```bash
python predict.py --checkpoint runs/mvanet_3/run_01/best.pth --input image.jpg
```

Predict one jujube with the paper's five-image voting scheme:

```bash
python predict.py --checkpoint runs/mvanet_3/run_01/best.pth --input ./jujube_001 --voting
```

Export ONNX:

```bash
python train.py export --checkpoint runs/mvanet_3/run_01/best.pth --verify
```

Count FLOPs/MACs:

```bash
python count_flops.py --input-size 32
```

## Output Artifacts

Each run is saved under:

```text
runs/<model_name>/run_XX/
```

Important files:

- `config.yaml`: resolved training config
- `environment.json`: Python/PyTorch/platform/Git metadata
- `dataset_summary.json`: split sizes and class counts
- `best.pth`, `last.pth`, `epoch_N.pth`: checkpoints with RNG state
- `run_XX_history.json`: epoch metrics
- `results.json`: test metrics for the best checkpoint
- `training_curves.png`, confusion matrices, per-class metric plots
- `../multirun_summary.json`: aggregate summary across repeated runs

## Resume

```bash
python train.py train --resume runs/mvanet_3/run_01/last.pth --data-root ./dataset
```

Resume restores model, optimizer, scheduler, and RNG state. Exact bitwise resume
still depends on the same PyTorch version, hardware/backend, and driver stack.
Resume is intentionally single-run only: do not combine `--resume` with
`--seed-mode five`, multiple `--seeds`, or `--num-runs > 1`. The checkpoint seed
must match the requested seed.
