# GhostNetV2 + JU-Lite-Mobile-Att trên STM32H750

Tài liệu này mô tả gói bổ sung 20 checkpoint (2 kiến trúc × 2 dataset × 5 seed)
đã được ghép vào pipeline TFLite Micro/CMSIS-NN hiện có.

## Trạng thái đã xác minh

- Audit checkpoint: **20/20 OK**; đúng metadata dataset, seed và classifier 6/4 lớp;
  không còn checkpoint trùng hash chéo dataset.
- Export ONNX tĩnh: **20/20 OK**, input `1×3×64×64`, output 6 hoặc 4 lớp.
- PTQ INT8 TFLite: **20/20 OK**, representative set cố định 512 ảnh lấy từ
  train+val; input/output float32, lõi graph INT8 NHWC.
- Golden PC trên toàn bộ test pack: **20/20 OK**.
- Op-set TFLite đã khớp resolver firmware: 14 op thực tế. JU-Lite được rewrite
  `HardSigmoid` để không sinh `RELU_0_TO_1`; runtime dùng `HARD_SWISH` có kernel TFLM.
- XIP firmware 16-op build sạch.

Kích thước linker của firmware benchmark:

| Vùng | Đã dùng | Dung lượng |
|---|---:|---:|
| Internal flash | 41.592 B | 128 KiB |
| QSPI app (TFLM XIP) | 88.032 B | 2 MiB |
| DTCM | 109.464 B | 128 KiB |
| TFLM arena (RAM_D1) | 504 KiB | 512 KiB |

Model lớn nhất là GhostNetV2 5.972.800 B, vẫn dưới slot QSPI 6 MiB
(6.291.456 B). JU-Lite khoảng 2,73 MB.

Golden macro-F1 trung bình 5 seed:

| Dataset | GhostNetV2 | JU-Lite-Mobile-Att |
|---|---:|---:|
| `jujube` (6 lớp, 1.740 test) | 0,8627 | 0,9236 |
| `jujube_ours` (4 lớp, 519 test) | 0,8772 | 0,9438 |

Chi tiết từng seed nằm trong `stm32/host/out/supplement/<dataset>/golden_tflite.csv`.

## Cấu trúc đầu ra

```text
stm32/host/out/supplement/
  checkpoint_audit.csv
  export_onnx.csv
  jujube/
    onnx/  tflite/  golden_tflite.csv  golden_summary.csv  tflite_ops.txt
    SD/manifest.csv  SD/models/*.tflite  SD/test/test_data.pack
  jujube_ours/
    onnx/  tflite/  golden_tflite.csv  golden_summary.csv  tflite_ops.txt
    SD/manifest.csv  SD/models/*.tflite  SD/test/test_data.pack
  firmware/
    benchmark/{jujube_h750.elf,jujube_h750_flash.hex,tflm_app.bin,...}
    ram_only/{jujube_h750.elf,jujube_h750_flash.hex,tflm_app.bin,...}
```

Hai thư mục `SD` là hai bộ độc lập. Không trộn manifest/model của dataset 6 lớp
với test pack 4 lớp (hoặc ngược lại).

## Tạo lại toàn bộ gói

Từ root project:

```powershell
python stm32\host\prepare_supplement.py all --max-calib 512
powershell -ExecutionPolicy Bypass -File stm32\cube\build_supplement.ps1
powershell -ExecutionPolicy Bypass -File stm32\cube\build_supplement.ps1 -RamOnly
```

Các stage có thể chạy riêng: `audit`, `export`, `pack`, `convert`, `golden`, `sd`.
Mặc định script dùng checkpoint ở `~/Downloads/bổ sung`; đổi bằng `--source` nếu cần.

## Chép thẻ và chạy board

1. Format thẻ FAT32, chọn **một** dataset rồi chép toàn bộ nội dung bên trong thư mục
   `SD` tương ứng vào root thẻ. Root phải có `manifest.csv`, `models/`, `test/`, `results/`.
2. Nạp code TFLM XIP vào QSPI app slot:

```powershell
python stm32\host\flash_tflm_to_qspi.py `
  stm32\host\out\supplement\firmware\benchmark\jujube_h750.elf
```

3. Nạp bootstrap internal flash và đọc log RAM qua ST-Link:

```powershell
powershell -ExecutionPolicy Bypass -File stm32\cube\flash_and_log.ps1 `
  -Elf stm32\host\out\supplement\firmware\benchmark\jujube_h750.elf `
  -WaitSeconds 600
```

Kết quả được append vào `/results/f1_latency.csv` trên thẻ, chỉ gồm Macro-F1,
latency trung bình và latency p95 cho từng seed.

### Kiểm tra arena trước khi benchmark

Để chỉ chạy `AllocateTensors()` và ghi `/results/ram_usage.csv`, dùng bộ firmware
`firmware/ram_only` trong cả hai lệnh flash phía trên. Đây là kiểm tra phần cứng quyết định
cho dung lượng arena 504 KiB; ước lượng PC không thay thế được allocator thật của TFLM.
