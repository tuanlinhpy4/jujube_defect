# MvANet — Multi-vision Attention Network

Hệ thống phân loại táo tàu (jujube) bằng mạng MvANet, hỗ trợ **train / eval / export** qua CLI duy nhất.

## 📁 Cấu trúc project

```
MvANet/
├── configs/
│   └── default.yaml           # Config mặc định (mọi flag CLI đều override được)
├── data/
│   ├── __init__.py
│   └── dataset.py             # DataLoader, transforms
├── models/
│   ├── __init__.py            # Dynamic registry — tự quét subfolder
│   ├── MvANet/                # Model gốc (PReLU + SE + Conv3x3)
│   │   ├── __init__.py
│   │   ├── blocks.py
│   │   └── mvanet.py
│   └── MvANet_opt/            # Model tối ưu (ReLU + ECA + Ghost + DropPath)
│       ├── __init__.py
│       ├── blocks.py
│       └── mvanet.py
├── utils/
│   ├── __init__.py
│   ├── logger.py
│   ├── metrics.py
│   └── visualize.py
├── train.py                   # CLI entry point chính
├── eval.py                    # Eval script standalone
├── export_onnx.py             # Export script standalone
├── predict.py                 # Inference script
├── count_flops.py             # So sánh FLOPs giữa các model
├── visualize_features.py      # Visualization feature maps
├── requirements.txt
└── README.md
```

## 🚀 Sử dụng nhanh

### 1. Cài đặt

```bash
pip install -r requirements.txt
```

### 2. Chuẩn bị dataset

```
dataset/
├── train/
│   ├── invalid/
│   ├── rotten/
│   ├── wizened/
│   └── normal/
├── val/
│   └── ...
└── test/
    └── ...
```

### 3. Xem danh sách model

```bash
python train.py info
```

### 4. Train

```bash
# Train với default config
python train.py train --data-root ./dataset

# Train model cụ thể
python train.py train --arch MvANet --model mvanet_2 --data-root ./dataset

# Train model tối ưu
python train.py train --arch MvANet_opt --model mvanet_1_tiny --data-root ./dataset

# Tùy chỉnh hyperparameters
python train.py train \
    --arch MvANet \
    --model mvanet_2 \
    --data-root ./dataset \
    --epochs 100 \
    --lr 0.01 \
    --optimizer adam \
    --lr-scheduler cosine \
    --batch-size 128 \
    --input-size 96

# Dùng file config khác
python train.py train --config configs/my_config.yaml

# Resume training
python train.py train --resume runs/mvanet_2/run_01/last.pth --data-root ./dataset
```

### 5. Đánh giá

```bash
python train.py eval \
    --checkpoint runs/mvanet_2/run_01/best.pth \
    --data-root ./dataset \
    --visualize
```

### 6. Export ONNX

```bash
python train.py export \
    --checkpoint runs/mvanet_2/run_01/best.pth \
    --verify
```

## 🔌 Thêm model mới

Chỉ cần **3 bước**:

1. Tạo folder `models/MyNewModel/`
2. Đặt code vào đó (blocks.py, mvanet.py, hoặc tên bất kỳ)
3. Tạo `__init__.py` export 2 symbol:

```python
# models/MyNewModel/__init__.py
from .mvanet import build_model, MODEL_CONFIGS

# MODEL_CONFIGS phải là dict: {"variant_name": {config_dict}, ...}
# build_model(name, num_classes, in_channels) phải trả về nn.Module có method count_parameters()
```

Sau đó dùng ngay:

```bash
python train.py train --arch MyNewModel --model variant_name
```

## 📋 Config Reference

Tất cả tham số trong `configs/default.yaml` đều có thể override bằng CLI flag:

| YAML key | CLI flag | Default | Mô tả |
|---|---|---|---|
| `model.arch` | `--arch` | MvANet | Tên folder architecture |
| `model.name` | `--model` | mvanet_2 | Tên variant |
| `model.num_classes` | `--num-classes` | 4 | Số lớp |
| `data.root` | `--data-root` | ./dataset | Thư mục dataset |
| `data.input_size` | `--input-size` | 32 | Resolution |
| `data.batch_size_train` | `--batch-size` | 64 | Batch size train |
| `train.epochs` | `--epochs` | 300 | Số epoch |
| `train.optimizer` | `--optimizer` | sgd | sgd / adam / amsgrad |
| `train.lr` | `--lr` | 0.001 | Learning rate |
| `train.lr_scheduler` | `--lr-scheduler` | plateau | plateau / cosine / step |
| `train.use_class_weights` | `--use-class-weights` | true | Dùng class weights |
| `train.num_runs` | `--num-runs` | 1 | Số lần train lặp |
| `train.seed` | `--seed` | 42 | Random seed |
| `train.save_dir` | `--save-dir` | ./runs | Thư mục lưu kết quả |
| `train.save_every` | `--save-every` | 50 | Lưu checkpoint mỗi N epoch |
| `train.resume` | `--resume` | null | Resume từ checkpoint |

## 🖥️ Sử dụng trên Google Colab

```python
# Cell 1: Mount Drive & setup
from google.colab import drive
drive.mount('/content/drive')

# Cell 2: Clone/copy project
!cp -r /content/drive/MyDrive/DATN/MvANet /content/MvANet
%cd /content/MvANet
!pip install -q -r requirements.txt

# Cell 3: Xem models
!python train.py info

# Cell 4: Train
!python train.py train \
    --arch MvANet \
    --model mvanet_2 \
    --data-root /content/dataset \
    --save-dir /content/runs \
    --epochs 300 --lr 0.001

# Cell 5: Eval
!python train.py eval \
    --checkpoint /content/runs/mvanet_2/run_01/best.pth \
    --data-root /content/dataset

# Cell 6: Export
!python train.py export \
    --checkpoint /content/runs/mvanet_2/run_01/best.pth \
    --verify
```
