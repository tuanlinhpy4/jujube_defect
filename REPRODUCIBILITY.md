# Reproducibility Checklist

This checklist maps the repository to common reviewer expectations for
research-code artifacts.

## Current Status

- Dependencies: core dependencies are in `requirements.txt`; optional ONNX/FLOPs
  dependencies are in `requirements-optional.txt`; tested versions are recorded
  in `requirements-lock.txt`.
- Entrypoints: `train.py train`, `train.py eval`, `train.py export`,
  `predict.py`, `visualize_features.py`, and `count_flops.py`.
- Exact seeds: single-seed mode uses `42`; five-seed mode uses
  `42, 123, 456, 789, 2024`.
- Train/test workflow: train mode automatically evaluates the best checkpoint
  on `dataset/test/` after every seed and writes `results.json`.
- Multi-seed report: `multirun_summary.json` contains mean +/- std for test
  error and test accuracy across seeds.
- Determinism: Python, NumPy, PyTorch, CUDA, and DataLoader worker seeds are
  configured; checkpoint resume stores RNG state.
- Resume safety: checkpoint resume is restricted to one seed/run at a time, and
  the checkpoint seed must match the requested seed.
- Data integrity: each run writes `dataset_summary.json` with split sizes and
  class counts.
- Class mapping: class order is fixed as `invalid, rotten, wizened, normal`.
- Experiment metadata: each run writes `config.yaml` and `environment.json`.
- Evaluation-only mode: `train.py eval` requires `--checkpoint`; it is intended
  only for existing weights and can run from `test/` or `val/` without requiring
  `train/`.

## Exact Commands

Single seed:

```bash
python train.py train --data-root ./dataset
```

Five seeds:

```bash
python train.py train --data-root ./dataset --seed-mode five
```

Evaluate existing weights:

```bash
python train.py eval --checkpoint runs/mvanet_3/run_01/best.pth --data-root ./dataset
```

## Notes For Submission

- Include or link the dataset and pretrained weights when the venue requires
  fully executable artifacts.
- If dataset-specific normalization is used, put exact values in
  `configs/default.yaml`; otherwise the default uses `ToTensor()` only.
- Resume each seed independently; never resume one checkpoint into a five-seed
  aggregate run.
- Exact bitwise reproducibility still depends on the same PyTorch version,
  hardware/backend, and driver stack.
