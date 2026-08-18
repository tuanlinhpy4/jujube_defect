# Đánh giá chuyển đổi GhostNetV2 FP32/PTH sang INT8 TFLite

Ngày đánh giá: 2026-07-12. Input `1×3×64×64`, TFLite lõi INT8, IO float32.

## Mã và artifact đã lưu

- Batch audit/export/convert: `stm32/host/prepare_supplement.py`
- Rewrite graph GhostNetV2: `stm32/host/prepare_ghostnetv2_onnx.py`
- ONNX → INT8 TFLite: `stm32/host/onnx_to_tflite.py`
- Đối chứng tự động PTH/ONNX/TFLite: `stm32/host/analyze_ghostnet_int8.py`
- Báo cáo từng seed: `stm32/host/out/supplement/ghostnet_int8_analysis.csv`
- Báo cáo Markdown sinh tự động: `stm32/host/out/supplement/ghostnet_int8_analysis.md`

Lệnh tái lập phân tích:

```powershell
stm32\host\.venv_tflite\Scripts\python.exe `
  stm32\host\analyze_ghostnet_int8.py
```

## Kết quả đối chứng

| Dataset | Seed | PTH F1 | ONNX raw | ONNX MCU | INT8 F1 | Giảm INT8 |
|---|---:|---:|---:|---:|---:|---:|
| jujube | 1 | 0,9142 | 0,9142 | 0,9142 | 0,8891 | -2,51 pp |
| jujube | 2 | 0,9182 | 0,9182 | 0,9182 | 0,8188 | -9,93 pp |
| jujube | 3 | 0,9132 | 0,9132 | 0,9132 | 0,8892 | -2,41 pp |
| jujube | 4 | 0,9219 | 0,9219 | 0,9219 | 0,8885 | -3,35 pp |
| jujube | 5 | 0,9181 | 0,9181 | 0,9181 | 0,8280 | -9,01 pp |
| jujube_ours | 1 | 0,9454 | 0,9454 | 0,9454 | 0,8095 | -13,59 pp |
| jujube_ours | 2 | 0,9397 | 0,9397 | 0,9397 | 0,9188 | -2,10 pp |
| jujube_ours | 3 | 0,9404 | 0,9404 | 0,9404 | 0,9133 | -2,71 pp |
| jujube_ours | 4 | 0,9522 | 0,9522 | 0,9522 | 0,8293 | -12,29 pp |
| jujube_ours | 5 | 0,9514 | 0,9514 | 0,9514 | 0,9152 | -3,62 pp |

Trung bình 5 seed:

- `jujube`: PTH `0,9171±0,0031`; INT8 `0,8627±0,0322`; giảm `5,44±3,32 pp`.
- `jujube_ours`: PTH `0,9458±0,0053`; INT8 `0,8772±0,0477`; giảm `6,86±5,00 pp`.

ONNX raw và ONNX đã rewrite có prediction agreement 100%. Sai lệch logits lớn nhất chỉ
`9,5e-6` đến `3,1e-5`. Do đó export, đổi layout và rewrite graph không gây tụt F1.

Prediction agreement giữa ONNX FP32 và INT8 chỉ còn:

- `jujube`: trung bình `0,883±0,042`; seed xấu 2/5 chỉ khoảng 0,832.
- `jujube_ours`: trung bình `0,909±0,043`; seed xấu 1/4 khoảng 0,852–0,863.

## Thử nghiệm số lượng calibration

PTQ chính dùng 512 ảnh train+val, seed chọn ảnh 42. Phân bố:

- `jujube`: 85/99/98/81/83/66 ảnh theo sáu lớp.
- `jujube_ours`: cracked 91, dry 212, insect_damaged 138, invalid 71.

Hai seed xấu được convert lại bằng toàn bộ train+val:

| Dataset/seed | 512 ảnh | Toàn bộ calibration | Thay đổi |
|---|---:|---:|---:|
| jujube_ours seed 1 | 0,8095 | 0,8103 (2.940 ảnh) | +0,08 pp |
| jujube seed 2 | 0,8188 | 0,7760 (9.855 ảnh) | -4,28 pp |

Tăng số ảnh không giải quyết vấn đề. Với min/max PTQ, thêm outlier còn mở rộng scale và làm
mất độ phân giải activation, nên seed 2 của bộ 6 lớp giảm thêm.

## Nguyên nhân chính

GhostNetV2 dùng DFC attention lặp lại nhiều lần. Graph TFLite của mỗi model có:

- 14 `LOGISTIC` + 14 `RESIZE_NEAREST_NEIGHBOR` cho gate DFC;
- 28 `MUL` cho gate DFC, SE và các tích feature;
- activation INT8 lượng tử theo tensor, trong khi trọng số convolution được per-channel.

Các gate sigmoid nằm trong `[0,1]` sau đó nhân liên tiếp vào feature map. Sai số scale/zero-point
ở gate và feature bị tích lũy qua nhiều `MUL`. Activation của các kênh Ghost/DFC có range rất khác
nhau nhưng phải chia sẻ một scale per-tensor; kênh biên độ nhỏ dễ bị làm tròn về zero. Phân bố
activation cũng thay đổi mạnh theo seed, giải thích vì sao cùng một calibration set nhưng có seed
chỉ giảm 2–3 pp và có seed giảm 9–14 pp.

Đây là độ nhạy PTQ của kiến trúc attention, không phải lỗi checkpoint, export hay rewrite.

## Hướng xử lý

1. **Ưu tiên QAT INT8** cho GhostNetV2: fake-quant cả convolution và đường DFC
   `sigmoid → resize → mul`, fine-tune từng seed rồi export TFLite.
2. Nếu chưa QAT, thử quantization debugger + clipping activation theo percentile/bias correction;
   không chỉ tăng số lượng ảnh calibration.
3. Phương án mixed precision: giữ DFC `LOGISTIC/MUL` ở float hoặc int16, conv vẫn INT8. Cần kiểm
   tra lại TFLM op coverage, arena và latency; sẽ mất lợi thế CMSIS-NN thuần INT8.
4. Không chọn riêng seed tốt dựa trên test để báo cáo. Vẫn chạy đủ 5 seed và báo mean±std.

## Kiểm tra STM32 đang chạy

GhostNetV2 `jujube_ours` seed 1 trên STM32H750:

- arena: `434/504 KiB`;
- Macro-F1: `0,8174`;
- latency trung bình: `1.276.378 us` ≈ `1,276 s/ảnh`;
- p95: `1.277.878 us` ≈ `1,278 s/ảnh`.

Sai khác F1 nhỏ so với TFLite PC (`0,8095`) có thể đến từ rounding/kernel CMSIS-NN; cần xem đủ
5 seed trước khi kết luận mức chênh on-device trung bình.
