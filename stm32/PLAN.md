# STM32H750 — Hệ thống benchmark nhiều model AI INT8 (Plan & Handoff)

> **File này là tài liệu bàn giao.** Khi mở Claude Code (trên Windows hoặc nơi khác),
> đọc file này trước để nắm toàn bộ mục tiêu, các quyết định đã chốt, trạng thái hiện tại,
> và plan để làm tiếp. Mục [§9 Tiếp tục từ đâu](#9-tiếp-tục-từ-đâu-cho-phiên-claude-mới) là điểm vào.

Cập nhật lần cuối: 2026-06-21 (P4: đã ghép TFLM+CMSIS-NN vào `inference.cc`, build sạch — xem §8) ·
Repo: `jujube` (branch làm việc: `thai`; các model wrapper nằm ở branch `linh`).

---

## 1. Mục tiêu

Trên **STM32H750VBT6** (Cortex-M7 @400MHz), **test nhiều model AI INT8 trong một phiên chạy**, tất cả dùng
chung một bộ test data cố định. Khi đổi model **không nạp lại firmware thủ công**: MCU đọc model từ SD card →
nạp vào QSPI Flash → chạy inference → ghi accuracy/F1/latency ra file. Mục tiêu cuối là một bảng so sánh
nhiều kiến trúc (accuracy INT8 + latency thật trên MCU) cho bài phân loại táo tàu (jujube) 4 lớp, input 64×64.

Bài toán domain: phân loại lỗi quả táo tàu — 4 lớp **`cracked, dry, insect_damaged, invalid`** (thứ tự index
alphabet theo ImageFolder: cracked=0, dry=1, insect_damaged=2, invalid=3).

---

## 2. Phần cứng

| | |
|---|---|
| MCU | STM32H750VBT6, Cortex-M7 @400MHz, FPU, I/D-Cache, DWT cycle counter |
| Internal Flash | **128 KB** (nhỏ — yếu tố quyết định kiến trúc, xem §4) |
| SRAM | ~1 MB |
| QSPI Flash | 8 MB (chạy app XIP + chứa model hiện tại) |
| SPI Flash | 8 MB (dự phòng / chứa app khi cần) |
| SD card | microSD qua SDMMC (chứa toàn bộ model + test data + manifest + results) |
| Giao tiếp | UART (log), SDMMC (SD), QSPI (app + model slot) |

Môi trường dev STM32 **đầy đủ trên Windows** (CubeMX/CubeIDE + arm-none-eabi-gcc). WSL chỉ dùng cho phần
Python (train/quantize/host-tools). **Có board H750 + ST-Link** → flash thật, lặp qua log UART.

---

## 3. Các model — đã chọn & đã kiểm chứng

Mục tiêu chạy ~50% của 25 ứng viên. Đã viết tool pre-flight (`stm32/preflight/check_tflm_ops.py`) check
**op-coverage (TFLite-Micro/CMSIS-NN) + flash-fit INT8** trên ONNX export @64×64, 4 lớp. Kết quả: **13 model GO**.

Runtime đã chốt: **TFLite Micro + CMSIS-NN**; model lưu dạng **`.tflite`** (tự mô tả graph → một app interpreter
cố định chạy được mọi kiến trúc; raw weights `.bin` KHÔNG đủ vì 13 kiến trúc khác nhau).

### 13 model GO — tên registry để train (`python train.py --dataset jujube --model <name> --seed 1`)

| Model | `--model` | INT8 (đo thật) | Ghi chú |
|---|---|---|---|
| EfficientNet-Lite0 | `efficientnet_lite0` | 3.8 MB | chuẩn INT8 nhất |
| MobileNetV2 | `mobilenet_v2` | 2.6 MB | |
| MobileNetV3-Small | `mobilenet_v3_small` | 1.86 MB | hswish+SE OK |
| MobileNetV4-Conv-Small | `mobilenetv4_conv_small` | 2.79 MB | |
| MNASNet-05 | `mnasnet0_5` | 1.21 MB | |
| FBNet-C100 | `fbnetc_100` | 4.07 MB | |
| ShuffleNetV2-0.5 | `shufflenet_v2_x0_5` | 0.59 MB | |
| SqueezeNet-1.1 | `squeezenet1_1` | 0.84 MB | |
| RegNetX-002 | `regnetx_002` | 1.66 MB | |
| RepVGG-A0 | `repvgg_a0` | 7.16 MB | **phải reparam**; >6MB slot → cần xử lý layout QSPI |
| GhostNetV2-100 | `ghostnetv2_100` | 5.95 MB | DFC+sigmoid → theo dõi int8_drop |
| TinyNet-E | `tinynet_e` | 1.13 MB | swish/SE → theo dõi int8_drop |
| **JU-Lite-Mobile-Att-Micro** | **`ju_lite_mobile_att_micro`** | (đang train) | ⚠ phải dùng đúng biến thể `_micro` (0.55M), KHÔNG dùng alias trống `ju_lite_mobile_att` (=2.16M) |

Optional thêm nếu muốn đa dạng: `efficientnet_b0`, `spnasnet_100`, `starnet_s1`.

### Bị loại (KHÔNG port xuống MCU)
- **Transformer/ConvNeXt** — op không có kernel INT8 trên TFLM/CMSIS-NN (`Erf`/GELU, `LayerNorm`,
  `InstanceNorm`, attention): ConvNeXt-Atto/Femto, FastViT-T8, PoolFormer-S12, EfficientViT-M0 (còn không
  export nổi @64×64), JUFormer, EfficientFormer-L1.
- **ResNet18** — op sạch nhưng ~8–11MB > 8MB QSPI → quá to.

Wrapper của tất cả model trên đã có sẵn ở **branch `linh`**, thư mục `models/` (tên file = tên registry =
`ARCH_NAME`). Registry tự scan file có `ARCH_NAME` + `build_model` + `MODEL_CONFIGS` (xem `models/__init__.py`).

---

## 4. Quyết định kiến trúc (đã chốt)

1. **Runtime = TFLite Micro + CMSIS-NN**, model = **`.tflite`**.
2. **Giữ thiết kế QSPI model-slot của kế hoạch gốc.** (Ban đầu từng tính "nạp model thẳng vào SRAM" — SAI với
   bộ model này vì chúng 0.6–7MB > 1MB SRAM; chỉ ShuffleNet/SqueezeNet vừa SRAM. Nên model **phải** ở QSPI.)
3. **1 ảnh firmware module hóa** (không tách 2 binary + jump như doc gốc). "Kernel" và "app inference" là các
   module trong cùng một binary.
4. **Phase weight-independent (hiện tại) chạy code từ INTERNAL FLASH.**
   Lý do then chốt: H750 chỉ 128KB internal flash. Nếu chưa link TFLM vào thì firmware (HAL+FatFS+QSPI+logic)
   **vừa 128KB** → code chạy từ internal flash, QSPI chỉ là kho dữ liệu → **nạp/xóa QSPI slot an toàn** (vì
   CPU không fetch lệnh XIP từ chính con QSPI đang ghi). Toàn bộ phần XIP-từ-QSPI phức tạp được hoãn tới lúc
   thêm TFLM (đúng lúc cần weight).
5. **Host bake sẵn preprocessing** vào `test_data.pack` (MCU không tự làm resize/normalize float) → khớp
   bit-for-bit với pipeline eval, tránh lệch accuracy.

---

## 5. Trạng thái dự án

| Giai đoạn | Trạng thái |
|---|---|
| **P1 — Khả thi & chọn model** | ✅ XONG (tool pre-flight + 13 model GO + chốt runtime + xác nhận QSPI slot) |
| **P2 — Train + quantize** | ✅ Gần xong — **50 `.tflite` INT8** (10 arch × 5 seed) đã sinh (`out/SD/models/`) |
| **P3 — Firmware harness (STUB)** | ✅ XONG & verify thật trên board (M1→M5 end-to-end) — xem `cube/BUILD.md` |
| **P4 — Tích hợp TFLM + accuracy thật + XIP** | 🔄 ĐANG LÀM — TFLM+CMSIS-NN đã ghép vào `inference.cc`, build sạch; xem §8 |

P2 config: train ở **64×64**, dataset `jujube` (lưu ý: `configs/jujube.yaml` mặc định 224×224/6-class — phải
đảm bảo train đúng 64×64/4-class cho khớp target MCU; user đã tự lo phần này).

---

## 6. Dataset & preprocessing (facts để khớp trên MCU)

- Đường dẫn: `data_jujube/data_resize/64_split_seed42_70_15_15/{train,val,test}/<class>/*.png`
- 4 lớp: `cracked, dry, insect_damaged, invalid` (index alphabet 0..3).
- **Test split = 519 ảnh** (cracked 93, dry 221, insect_damaged 130, invalid 75), PNG RGB 64×64, ~2.7MB.
- **Eval transform (PHẢI tái tạo y hệt khi bake pack):**
  `Resize(int(64*256/224)=73) → CenterCrop(64) → ToTensor(/255) → Normalize(mean,std)`, thứ tự kênh **RGB**, layout **NCHW**.
  - `mean = [0.485, 0.456, 0.406]`, `std = [0.229, 0.224, 0.225]` (ImageNet; cũng nằm trong `configs/jujube*.yaml`).
  - ⚠ Dù ảnh đã 64×64, transform vẫn Resize→73→Crop→64 (KHÔNG identity) → phải bake đúng.
- Quantize hiện tại (tham chiếu `quantize_int8.py`, `junet_mcu_stm32/eval_int8.py`): ONNX static INT8, QDQ,
  per-channel, weight `qint8` / activation `quint8`, calib trên train split.

---

## 7. Plan Phase 3 — Kernel + harness (weight-independent)

### 7.1 Cấu trúc thư mục (sẽ tạo dưới `stm32/`)
```
stm32/
  PLAN.md              ← file này
  README.md            ← spec CubeMX + cách tích hợp + build/flash (sẽ viết)
  app/
    model_ctx.h        ← struct hợp đồng kernel↔app (slot addr, w/h/c, num_classes, paths)
    kernel.c/.h        ← init phần cứng + vòng lặp manifest + nạp model + dispatch
    board.c/.h         ← clock 400MHz, I/D-cache, MPU vùng QSPI, DWT init
    sdcard.c/.h        ← mount FatFS + helper đọc file
    manifest.c/.h      ← parse manifest.csv
    qspi_model.c/.h    ← erase/program QSPI model slot + CRC32 verify + memory-mapped
    test_pack.c/.h     ← reader test_data.pack (stream từng sample vào SRAM)
    inference.c/.h     ← infer(input,output): backend STUB now / TFLM sau
    metrics.c/.h       ← confusion matrix, accuracy, macro-F1, latency avg/p95
    results.c/.h       ← append results.csv qua FatFS
    crc32.c/.h
  host/
    make_test_pack.py  ← bake preprocessing → test_data.pack (KHÔNG cần weight)
    make_manifest.py   ← quét .tflite → crc32 → manifest.csv
    PACK_FORMAT.md     ← đặc tả nhị phân pack/manifest/results
```

### 7.2 Spec CubeMX (gen project trên Windows)
- MCU **STM32H750VBTx**; SYSCLK **400MHz** (HSE→PLL); bật I/D-Cache.
- **SDMMC1** 4-bit + middleware **FATFS**.
- **QUADSPI** đúng chân QSPI chip của board (memory-mapped mode).
- **USART** VCP của board (115200 8N1) cho log.
- FPU mặc định M7; DWT bật trong `board.c`; MPU cho vùng `0x90000000` (QSPI) set trong `board.c`.
- Claude sẽ cung cấp địa chỉ + các "USER CODE BEGIN/END" hook để gọi `kernel_main()` từ `main.c`.

### 7.3 Định dạng file trên SD
- **QSPI memory map** (theo doc gốc): `0x90000000–0x901FFFFF` = app cố định (2MB) · `0x90200000–0x907FFFFF`
  = current_model_slot (6MB). *Lưu ý RepVGG 7.16MB > 6MB → cần nới slot / dùng SPI flash thứ 2 / hoặc bỏ RepVGG.*
- `test_data.pack`: header `{magic, ver, N, C=3, H=64, W=64, dtype=float32}` + N×(`float32[3*64*64]` + `uint8 label`).
  N=519, ~25MB (vừa SD). App quantize float→int8 theo scale/zp của model lúc chạy (khi có TFLM).
- `manifest.csv`: `model_id,model_bin,input_w,input_h,input_c,num_classes,crc32`
- `results.csv`: `model_id,accuracy,macro_f1,latency_avg_ms,latency_p95_ms,num_samples`
- SD layout: `/manifest.csv`, `/models/*.tflite`, `/test/test_data.pack`, `/results/results.csv`

### 7.4 Hợp đồng kernel↔app (`model_ctx.h`)
Struct chứa: base addr model slot trong QSPI, input w/h/c, num_classes, đường dẫn test_data.pack & results.csv,
con trỏ buffer SRAM. Kernel điền → app đọc. (Trong 1 ảnh nó chỉ là ranh giới module sạch.)

### 7.5 Latency (DWT)
```c
CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;   // bật trong board_init
DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
uint32_t s = DWT->CYCCNT; /* infer */ uint32_t e = DWT->CYCCNT;
extern uint32_t SystemCoreClock;
float ms = (float)(e - s) / (float)SystemCoreClock * 1000.0f;
```
Lưu mảng latency 519 phần tử để tính avg + p95.

### 7.6 Milestones (mỗi mốc test được trên board, KHÔNG cần weight)
- **M1 — Init**: clock/cache/FPU/MPU/DWT + UART + SDMMC/FatFS + QSPI mem-mapped. DoD: mount SD, đọc 1 file, log version.
- **M2 — Manifest + nạp QSPI**: parse manifest.csv; copy blob SD→QSPI slot; CRC32 verify; read-back qua memory-mapped. DoD: nạp 1 blob dummy, CRC khớp.
- **M3 — Data path**: reader test_data.pack stream từng sample vào SRAM (~49KB/sample). DoD: đọc đủ 519, label hợp lệ.
- **M4 — Metrics + results**: confusion matrix/accuracy/macro-F1/p95 + ghi results.csv. DoD: feed logits tổng hợp → số đúng, CSV append OK.
- **M5 — Harness end-to-end (STUB infer)**: vòng lặp manifest → mỗi model: load pack → STUB infer (bọc DWT) → metrics → results.csv. DoD: chạy nhiều "model" dummy 1 phiên, in bảng tổng kết UART, latency ra số thật.

### 7.7 Host tools (chạy WSL ngay, không cần weight)
- `make_test_pack.py` → sinh `test_data.pack` từ test split (data đã có).
- `make_manifest.py` → sinh `manifest.csv` (giờ test với blob dummy; với `.tflite` thật sau).

---

## 8. Phase 4 — Tích hợp TFLM + CMSIS-NN (ĐANG LÀM)

### 8.1 Đã xong
- **Convert `.tflite` thật** (stage-2): host venv `host/.venv_tflite` (TensorFlow 2.21 + onnx2tf, `pip check` sạch).
  Pipeline `.pth → ONNX (export_onnx.py) → INT8 .tflite (onnx_to_tflite.py)` → **50 model** (10 arch × 5 seed)
  ở `out/SD/models/`. **IO float32, Quantize/Dequantize trong graph, activation/weight int8** (khớp CMSIS-NN).
- **Lấy nguồn TFLM** (máy CÓ mạng): clone `tflite-micro` vào `.tflm_clone/` (gitignore), tải thủ công các dep
  pinned vào `…/downloads/`: **flatbuffers v25.9.23, gemmlowp 719139ce, ruy d3712831, CMSIS-NN 6d9d61d8**.
  ⚠ **KHÔNG dùng CMSIS-NN trong pack H7** (`Drivers/CMSIS/NN`) — đó là API q7/q15 **cũ** (CMSIS-5), TFLM cần
  API `arm_*_s8` mới. (`create_tflm_tree.py` cần `make` — máy không có → tự ghép bằng CMake glob, xem dưới.)
- **Backend `inference.cc`** (thay `inference.c` STUB): đọc model tại `ctx->slot_base` (QSPI memmap) →
  `MicroMutableOpResolver` (HỢP op) → `MicroInterpreter`; input pack NCHW→NHWC float; Invoke; output logits.
  Arena **256KB ở RAM_D2** (section `.tflm_arena`, tách khỏi `.bss`/`s_input` ở RAM_D1).
- **Build C++ bare-metal**: `cube/tflm.cmake` (glob TFLM core + kernel + thay bản `cmsis_nn` + source CMSIS-NN),
  `project(... C CXX ASM)`, flags `-std=gnu++17 -fno-exceptions -fno-rtti -fno-threadsafe-statics`,
  `-DTF_LITE_STATIC_MEMORY -DCMSIS_NN`. `app/tflm_shim.cc`: `DebugLog→uart_log`, operator new (pool tĩnh, vì
  nosys không có `_sbrk`), `__cxa` stub. **Compile sạch toàn bộ** (~270 file).

### 8.2 Kết quả kích thước (đo thật, gnu-tools 14.3.1, -Os)
| Build | FLASH | Ghi chú |
|---|---|---|
| HỢP 13 op (toàn bộ 50 model) | **144 712 B / 128 KB = 110% → TRÀN 13 640 B** | ⇒ **bắt buộc QSPI XIP** |
| Chỉ 8 op của `mnasnet` (`-DMNASNET_ONLY=ON`) | **126 408 B / 128 KB = 96.4% → VỪA** | dùng test Bước A trên internal flash |

RAM: D1 ~105KB/512 (s_input 48 + cxx-pool + .bss), D2 256KB/288 (arena). **Flash là nút thắt, RAM dư.**

### 8.3 Đang dở / còn lại
- **Bước A — kiểm đúng (internal flash, mnasnet-only):** đã **flash** bản 8-op (96.4%), board chạy
  (SD mount, QSPI OK). NHƯNG **thẻ SD vẫn là layout Phase-3 (dummy `.bin`)** → cần copy `out/SD/manifest.csv`
  (hoặc `manifest_mnasnet.csv`) + `out/SD/models/mnasnet0_5_s*.tflite` lên thẻ (test_data.pack đã có) →
  chạy lại `flash_and_log.sh` → so accuracy on-device vs **golden mnasnet0_5 acc≈0.811 / F1≈0.801**.
- **Bước B — QSPI XIP (để chạy HỢP op + cả 50 model 1 phiên):** bootstrap nhỏ ở internal flash + app/TFLM XIP
  từ QSPI (`0x90000000`, 2MB đầu đã chừa trong `model_ctx.h`); **toàn bộ đường nạp QSPI** (abort→erase→program→
  re-enable memmap + HAL/log liên quan) phải chạy từ **RAM** (`.RamFunc`) vì không ghi được QSPI khi đang fetch
  lệnh XIP từ chính nó. Lúc *inference* thì OK (đọc weight memmap + fetch code XIP đều là read).
- **Op chưa hỗ trợ:** `RELU_0_TO_1` (chỉ `mobilenet_v3_small`, 5/50 model) **không có kernel TFLM** →
  cần re-export mv3 (thay clamp [0,1] bằng RELU6/khác) hoặc bỏ. Các arch còn lại op-set đủ.
- **Golden host** mới chạy 1/50 (`mnasnet0_5_s1`) → chạy nốt `eval_tflite_golden.py` cho cả 50 để có bảng tham chiếu.
- **Đã chạy benchmark on-board 2026-06-26:** SD `F:\results\results.csv` có **85/90** kết quả, mỗi dòng `N=519`,
  không trùng/không model lạ. Thiếu 5 seed `efficientnet_b0_s1..s5` vì TFLM `AllocateTensors FAIL`
  với arena hiện tại **504 KB**; QSPI load/CRC và test pack đều OK trước điểm fail.
- Bảng so sánh đã lưu trong repo:
  `stm32/host/out/benchmark_table_sd_clean.md` (Markdown đọc được) và
  `stm32/host/out/benchmark_table_sd.csv` (CSV). Top nhanh: `jel_slim_rep` 279.025 ms, F1 0.9500;
  cân bằng tốt: `jel_slim` 296.303 ms, F1 0.9531; F1 cao nhất: `squeezenet1_1` 0.9582 nhưng 917.073 ms.

### 8.4 Tái tạo nguồn TFLM (nếu xoá `.tflm_clone/`)
```bash
cd stm32 && git clone --depth 1 https://github.com/tensorflow/tflite-micro.git .tflm_clone
# tải 4 dep pinned (flatbuffers/gemmlowp/ruy/cmsis_nn) vào
#   .tflm_clone/tensorflow/lite/micro/tools/make/downloads/  (xem commit trong §8.1)
bash cube/build.sh                         # build HỢP 13-op (sẽ báo tràn flash -> chờ XIP)
cmake ... -DMNASNET_ONLY=ON && ninja ...   # build 8-op vừa flash (test Bước A)
```

---

## 9. Tiếp tục từ đâu (cho phiên Claude mới)

**Trạng thái: P4 benchmark on-board đã chạy gần đủ.** Firmware TFLM/XIP đọc SD, nạp QSPI, infer và ghi
`results.csv` được **85/90** model-seed. `efficientnet_b0_s1..s5` fail tại `AllocateTensors` do arena 504 KB,
không phải lỗi SD/QSPI. Bảng tổng hợp nằm ở `stm32/host/out/benchmark_table_sd_clean.md` và CSV ở
`stm32/host/out/benchmark_table_sd.csv`. Việc kế tiếp theo thứ tự:

1. Quyết định xử lý `efficientnet_b0`: bỏ khỏi bảng chính, giảm model, hoặc tăng/đổi bố trí arena nếu cần chạy đủ 90/90.
2. Chạy nốt **golden host** (`eval_tflite_golden.py`) cho cùng tập model để có cột đối chiếu PC vs STM32.
3. Xử lý `RELU_0_TO_1` cho `mobilenet_v3_small` nếu muốn đưa nhóm này vào benchmark đầy đủ (re-export hoặc bỏ).
4. Nếu cần báo cáo, dùng bảng hiện có: F1 cao nhất `squeezenet1_1`; nhanh nhất OK `jel_slim_rep`; cân bằng tốt `jel_slim`.

**Ràng buộc/ghi nhớ:**
- Build C++: dùng `cube/tflm.cmake`; KHÔNG đụng CMSIS-NN trong pack (API q7 cũ). Nguồn TFLM ở `.tflm_clone/`
  (gitignore) — tái tạo theo §8.4. `inference.h` giữ nguyên interface → `kernel.c` không đổi.
- Flash 128KB là nút thắt: chỉ build mnasnet-only mới vừa; HỢP op phải đợi XIP.
- Log đọc qua **ST-Link RAM mailbox** (`flash_and_log.sh`) vì board không có VCP. DebugLog của TFLM cũng vào
  mailbox này (bắt lỗi arena/op rất tiện).

**Tham chiếu trong repo:**
- `stm32/preflight/check_tflm_ops.py` + `stm32/preflight/out/preflight_report.csv` — tool & kết quả pre-flight.
- `Kế hoạch STM32 cho hệ thống test AI INT8.docx` — kế hoạch gốc tiếng Việt (kernel/app split, memory map, schemas).
- `models/` (branch `linh`) — wrapper 13 model; `models/__init__.py` — cơ chế registry.
- `configs/jujube*.yaml`, `data/dataset.py` — transform/normalize; `quantize_int8.py`, `junet_mcu_stm32/eval_int8.py` — pipeline INT8.
- `train.py` — CLI: `--dataset {jujube,jujube_ours} --model <name> --seed <ints>`.
- `IMX93/` — firmware NXP i.MX93 M33 (chỉ tham khảo pattern FSM/timing, KHÔNG dùng lại HAL cho H750).
