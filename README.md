# Jujube Defect Classification

Lightweight pretrained CNN benchmark for jujube quality grading.  
Targets reproducible A\*/Q1 reporting: fixed 70/15/15 stratified split, 5-seed protocol, mean ± std on test.

## Datasets

| Dataset | Classes | Split seed |
|---------|---------|------------|
| `jujube` | niaozhuo, meibian, liekou, haozao, gantiao, bianxing (6) | 42 |
| `jujube_ours` | dry, insect_damaged, cracked, mold, invalid (5) | 42 |
| `fruitvision` | 5 fruits × {Fresh, Rotten, Formalin-mixed} (15) | 42 |
| `sapota` | Large, Medium, Small, Spoiled (4) | 42 |
| `jujube_healthy` | bruised, healthy (2) | 42 |
| `mango_dhds` | Anthracnose, Bacterial Canker, Healthy, Scab, Stem End Rot (5) | 42 |

`sapota`, `jujube_healthy`, and `mango_dhds` come from the same Hugging Face
repo as `fruitvision` (https://huggingface.co/datasets/doantrongthai/FruitVision),
as separate zip files. Their class folders are auto-detected at load time —
the yaml configs don't hardcode `class_names`/`num_classes`.

Datasets are **auto-downloaded** on first run. To download manually:
```bash
python download_data.py                     # all datasets
python download_data.py --dataset jujube
python download_data.py --dataset fruitvision
python download_data.py --dataset sapota
python download_data.py --dataset jujube_healthy
python download_data.py --dataset mango_dhds
```

Or place datasets manually at `data/jujube/`, `data/jujube_ours/` as flat class folders:
```
data/jujube/
  niaozhuo/  meibian/  liekou/  haozao/  gantiao/  bianxing/
data/jujube_ours/
  dry/  insect_damaged/  cracked/  mold/  invalid/
```

`fruitvision` ships as a nested `Fruit/Condition` layout (e.g. `Apple/Fresh/`, `Apple/Rotten/`, `Apple/Formalin-mixed/`).
`download_data.py` flattens this into 15 combined class folders automatically:
```
data/fruitvision/
  Apple_Fresh/  Apple_Rotten/  Apple_Formalin-mixed/
  Banana_Fresh/ Banana_Rotten/ Banana_Formalin-mixed/
  Grape_Fresh/  Grape_Rotten/  Grape_Formalin-mixed/
  Mango_Fresh/  Mango_Rotten/  Mango_Formalin-mixed/
  Orange_Fresh/ Orange_Rotten/ Orange_Formalin-mixed/
```
Source: https://huggingface.co/datasets/doantrongthai/FruitVision

`sapota`, `jujube_healthy`, `mango_dhds` are already flat class folders after
extraction — no reorganizing needed, classes are picked up automatically:
```
data/sapota/
  Large/  Medium/  Small/  Spoiled/
data/jujube_healthy/
  bruised/  healthy/
data/mango_dhds/
  Anthracnose/  "Bacterial Canker"/  Healthy/  Scab/  "Stem End Rot"/
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

## Loss

Cross-entropy with **inverse-frequency class weights** (auto-computed from training split):

```
w_c = N / (C × n_c)
```

## Setup

```bash
pip install -r requirements.txt
pip install gdown   # for auto-download
```

## Training

```bash
python train.py --dataset jujube --model mobilenet_v3_small --seed 1

python train.py --dataset jujube_ours --model efficientnet_b0 --seed 1 2 3 4 5

python train.py --dataset jujube --model resnet18 --seed 1 2 3

python train.py --dataset sapota --model mobilenet_v3_small --seed 1 2 3 4 5

python train.py --dataset jujube_healthy --model mobilenet_v3_small --seed 1 2 3 4 5

python train.py --dataset mango_dhds --model efficientnet_b0 --seed 1 2 3 4 5
```

`--seed` accepts any number of values. Each seed is run sequentially.  
After all seeds complete, mean ± std is printed and saved to `runs/<dataset>/<model>/cross_entropy/summary.json`.

## GradCAM Visualization

Run after training to generate heatmap figures for the paper:

```bash
python visualize.py \
    --weights runs/jujube/mobilenet_v3_small/cross_entropy/seed_1/best.pth \
    --data-root data/jujube \
    --output-dir runs/jujube/mobilenet_v3_small/cross_entropy/seed_1/gradcam

# Multiple samples per class
python visualize.py \
    --weights runs/jujube/resnet18/cross_entropy/seed_1/best.pth \
    --data-root data/jujube \
    --n-samples 3 \
    --output-dir runs/jujube/resnet18/cross_entropy/seed_1/gradcam
```

Output: `gradcam_grid.png` — one column per class showing original image and GradCAM overlay.  
Prediction label shown in **green** (correct) or **red** (wrong).

## Reproducibility

- **Split**: stratified 70/15/15, always seed=42, fixed across all runs.
- **Training seed**: controls model head initialization and data loading order only.
- **Pretrained backbone**: fixed ImageNet weights, unaffected by seed.
- **Deterministic mode**: enabled via `torch.use_deterministic_algorithms`.
- **Class weights**: computed once from the fixed training split, identical across seeds.
- Each run saves `best.pth`, `history.json`, `test_results.json`, confusion matrix, training curves.

## Output Structure

```
runs/
  <dataset>/
    <model>/
      cross_entropy/
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
          gradcam/
            gradcam_grid.png
        seed_2/ ...
```

## Reported Metrics

Test set (per seed and mean ± std):

| Metric | Description |
|--------|-------------|
| Accuracy | Overall correct / total |
| Macro F1 / P / R | Unweighted average across classes |
| Weighted F1 / P / R | Support-weighted average |
| Cohen's κ | Agreement beyond chance (×100) |
| MCC | Matthews Correlation Coefficient (×100) |
| AUC-ROC macro | One-vs-rest macro AUC (×100) |
| AUC-ROC per class | Per-class one-vs-rest AUC |

Multi-seed: mean ± std (ddof=1) for Accuracy, F1, P, R, κ, MCC, AUC-ROC.