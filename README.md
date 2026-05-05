# Jujube Defect Classification

Lightweight pretrained CNN benchmark for jujube quality grading.  
Targets reproducible A\*/Q1 reporting: fixed 70/15/15 stratified split, 5-seed protocol, mean ± std on test.

## Datasets

| Dataset | Classes | Split seed |
|---------|---------|------------|
| `jujube` | niaozhuo, meibian, liekou, haozao, gantiao, bianxing (6) | 42 |
| `jujube_ours` | dry, insect_damaged, cracked, mold, invalid (5) | 42 |

Place datasets at `data/jujube/` and `data/jujube_ours/` as flat class folders:
```
data/jujube/
  niaozhuo/  meibian/  liekou/  haozao/  gantiao/  bianxing/
data/jujube_ours/
  dry/  insect_damaged/  cracked/  mold/  invalid/
```

## Models

One file per model family under `models/`. All use ImageNet pretrained weights.

| Model | Params (M) | Family |
|-------|-----------|--------|
| `mobilenet_v3_small` | 2.54 | MobileNetV3 |
| `mobilenet_v2` | 3.50 | MobileNetV2 |
| `efficientnet_b0` | 5.29 | EfficientNet |
| `shufflenet_v2_x0_5` | 1.37 | ShuffleNetV2 |
| `mnasnet0_5` | 2.22 | MNASNet |
| `squeezenet1_0` | 1.25 | SqueezeNet |
| `resnet18` | 11.69 | ResNet |

## Loss Functions

| Loss | File |
|------|------|
| `cross_entropy` | `loss/cross_entropy.py` |
| `focal` | `loss/focal.py` |
| `label_smoothing` | `loss/label_smoothing.py` |

## Setup

```bash
pip install -r requirements.txt
```

## Training

```bash
python train.py --dataset jujube --model mobilenet_v3_small --loss cross_entropy --seed 1

python train.py --dataset jujube_ours --model efficientnet_b0 --loss focal --seed 1 2 3 4 5

python train.py --dataset jujube --model resnet18 --loss label_smoothing --seed 1 2 3
```

`--seed` accepts any number of values. Each seed is run sequentially.  
After all seeds complete, mean ± std is printed and saved to `runs/<dataset>/<model>/<loss>/summary.json`.

## Reproducibility

- **Split**: stratified 70/15/15, always seed=42, fixed across all runs.
- **Training seed**: controls model head initialization and data loading order only.
- **Pretrained backbone**: fixed ImageNet weights, unaffected by seed.
- **Deterministic mode**: enabled via `torch.use_deterministic_algorithms`.
- Each run saves `best.pth`, `history.json`, `test_results.json`, confusion matrix, training curves.

## Output Structure

```
runs/
  <dataset>/
    <model>/
      <loss>/
        split_info.json
        summary.json          ← mean ± std across seeds
        environment.json
        seed_1/
          best.pth
          history.json
          test_results.json
          training_curves.png
          confusion_matrix.png
          per_class_metrics.png
        seed_2/ ...
```

## Reported Metrics

Test set: Accuracy, Macro-F1, Macro-Precision, Macro-Recall, Weighted-F1.  
Multi-seed: mean ± std (ddof=1) across all provided seeds.
