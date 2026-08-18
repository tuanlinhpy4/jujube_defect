# Định dạng — `test_data.pack`, `manifest.csv`, `f1_latency.csv`, `ram_usage.csv`

> Spec dùng chung host (Python, `make_test_pack.py` / `make_manifest.py`) ↔ firmware
> (`test_pack.c`, `manifest.c`, `results.c`). Mọi số nhị phân **little-endian** (khớp Cortex-M7).

---

## 1. `test_data.pack`

Chứa toàn bộ test split đã **preprocess sẵn** (Resize 73 → CenterCrop 64 → ToTensor /255 →
Normalize ImageNet), để MCU không phải tự resize/normalize → khớp bit-for-bit với eval PyTorch.
MCU đọc từng sample, rồi tự quantize float→int8 theo scale/zp của model lúc chạy (Phase 4).

### Header — 40 byte

| Offset | Field | Type | Giá trị |
|---|---|---|---|
| 0  | `magic`        | char[4] | `"JPK1"` (0x4A 0x50 0x4B 0x31) |
| 4  | `version`      | uint32  | `1` |
| 8  | `num_samples`  | uint32  | N = 519 |
| 12 | `channels`     | uint32  | C = 3 |
| 16 | `height`       | uint32  | H = 64 |
| 20 | `width`        | uint32  | W = 64 |
| 24 | `dtype`        | uint32  | `0` = float32 (1 = fp16, hiện không dùng) |
| 28 | `layout`       | uint32  | `0` = NCHW (channel-major, mỗi plane row-major) |
| 32 | `label_bytes`  | uint32  | `4` (label = uint32/sample) |
| 36 | `reserved`     | uint32  | `0` |

### Phần dữ liệu — N record, mỗi record `stride = C*H*W*4 + 4 = 49156` byte

```
record[i]:
  float32 image[C*H*W]   # NCHW: [R-plane 64x64][G-plane][B-plane], mỗi plane row-major
  uint32  label          # 0..3  (cracked=0, dry=1, insect_damaged=2, invalid=3)
```

- `image` đã normalize: `x = (pixel/255 - mean) / std`, `mean=[0.485,0.456,0.406]`,
  `std=[0.229,0.224,0.225]`, thứ tự kênh **RGB**.
- Label = uint32 (không phải uint8) để mỗi record căn 4 byte → đọc float không lệch alignment trên M7.
- Tổng kích thước ≈ `40 + 519*49156` ≈ **24.3 MB**.

### Đọc trên MCU (tham chiếu `test_pack.c`)
```c
seek(40 + i*49156); read(buf, 49152);        // image vào SRAM buffer (căn 4 byte)
seek(40 + i*49156 + 49152); read(&label, 4); // hoặc đọc cả 49156 một lần
```

---

## 2. `manifest.csv`

ASCII, có dòng header, ngăn cách bằng dấu phẩy. Một dòng = một model cần test.

```
model_id,model_bin,input_w,input_h,input_c,num_classes,crc32
mobilenet_v2,models/mobilenet_v2.tflite,64,64,3,4,a1b2c3d4
shufflenet_v2_x0_5,models/shufflenet_v2_x0_5.tflite,64,64,3,4,deadbeef
```

| Cột | Ý nghĩa |
|---|---|
| `model_id`     | tên hiển thị + key ghi `results.csv` (= tên file, không đuôi) |
| `model_bin`    | đường dẫn file model trên SD, **tương đối gốc SD** (vd `models/x.tflite`) |
| `input_w/h/c`  | 64 / 64 / 3 |
| `num_classes`  | 4 |
| `crc32`        | CRC-32/ISO-HDLC (zlib `crc32`, poly reflected 0xEDB88320) của file model, **8 hex chữ thường**, không `0x` |

MCU: nạp `model_bin` từ SD → QSPI slot → đọc lại memory-mapped → tính CRC32 → so với cột `crc32`.

---

## 3. `results/f1_latency.csv`

Firmware tạo (nếu chưa có ghi dòng header), rồi **append** mỗi model một dòng.

```
model_id,macro_f1,latency_avg_ms,latency_p95_ms
mobilenet_v2_s1,0.9034,12.473,15.880
```

| Cột | Ý nghĩa |
|---|---|
| `macro_f1`        | macro-averaged F1 (trung bình F1 của 4 lớp), 4 chữ số |
| `latency_avg_ms`  | trung bình thời gian infer 1 ảnh (DWT cycle / `SystemCoreClock`; 400MHz trong build báo cáo), 3 chữ số |
| `latency_p95_ms`  | percentile 95 latency, 3 chữ số |

---

## 4. `results/ram_usage.csv`

Firmware tạo file này khi build với `-DJUJUBE_RAM_ONLY=ON`. Chế độ này chỉ nạp model,
gọi `AllocateTensors()`, ghi RAM arena, rồi chuyển model tiếp theo; không mở
`test_data.pack` và không chạy inference trên tập test.

```
model_id,status,model_size_bytes,tflm_arena_used_bytes,tflm_arena_capacity_bytes,input_w,input_h,input_c,num_classes,note
jedgenet_s1,OK,449096,319120,516096,64,64,3,4,
```

| Cột | Ý nghĩa |
|---|---|
| `status` | `OK`, `LOAD_FAIL`, hoặc `INIT_FAIL` |
| `model_size_bytes` | kích thước file model đã nạp vào QSPI slot |
| `tflm_arena_used_bytes` | RAM arena TFLM thật sau `AllocateTensors()` |
| `tflm_arena_capacity_bytes` | kích thước arena build vào firmware, hiện 504 KiB |
| `input_w/h/c` | shape input từ `manifest.csv` |
| `num_classes` | số lớp từ `manifest.csv` |
| `note` | ghi chú lỗi ngắn nếu có |

---

## 5. CRC-32

Cả host và firmware dùng **CRC-32/ISO-HDLC** (chuẩn zlib / Ethernet):
- poly reflected `0xEDB88320`, init `0xFFFFFFFF`, xorout `0xFFFFFFFF`, input/output reflected.
- Host: `zlib.crc32(data) & 0xFFFFFFFF`. Firmware: `crc32.c` table-based khớp y hệt.

---

## 6. SD card layout
```
/manifest.csv
/models/<model_id>.tflite      (Phase 4; Phase 3 dùng blob dummy .bin để test harness)
/test/test_data.pack
/results/f1_latency.csv        (firmware tạo nếu chưa có)
/results/ram_usage.csv         (firmware tạo trong RAM-only mode)
```
