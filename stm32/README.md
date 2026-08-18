# STM32H750 jujube benchmark harness — build & integrate (Phase 3)

Firmware **weight-independent**: nạp model từ SD → QSPI slot → infer (backend **STUB** hiện tại)
→ ghi accuracy/F1/latency ra `results.csv`. Mục tiêu Phase 3 = chạy được toàn bộ pipeline với
blob model **dummy** (chưa cần weight/TFLM). Xem [PLAN.md](PLAN.md) cho bối cảnh, file này cho cách build.

Board: **WeAct MiniSTM32H7xx** (STM32H750VBT6, HSE 25MHz, W25Q64 QSPI 8MB, SD qua SDMMC1).

---

## 0. Cấu trúc
```
stm32/
  app/        ← code C của harness (thêm vào project CubeIDE)
  host/       ← tool Python (chạy trên PC, sinh pack/manifest)
  out/        ← (gitignore) pack + SD layout sinh ra để test
```

---

## 1. Host: sinh dữ liệu lên SD (chạy trước, trên PC)

```bash
# 1) test_data.pack từ test split (float32, bit-exact eval transform)
python stm32/host/make_test_pack.py \
    --test-dir data_jujube/data_resize/64_split_70_15_15_seed42/test \
    --out stm32/out/SD/test/test_data.pack

# 2) manifest.csv từ thư mục model. Phase 3 dùng blob .bin dummy để test harness:
python stm32/host/make_manifest.py \
    --models-dir stm32/out/SD/models --ext .bin \
    --sd-subdir models --out stm32/out/SD/manifest.csv
#    Phase 4: --ext .tflite (model thật)
```

**Copy lên thẻ SD (FAT32)** đúng layout:
```
/manifest.csv
/models/<model>.bin        (Phase 4: .tflite)
/test/test_data.pack
/results/                  (firmware tự tạo file results.csv)
```

> ⚠ Tên file dài (`test_data.pack`, `mobilenet_v2.tflite`) **không phải 8.3** → BẮT BUỘC bật
> Long File Name trong FatFS (xem §3).

---

## 2. CubeMX — tạo project (STM32H750VBTx)

| Mục | Cấu hình |
|---|---|
| **RCC** | HSE = **Crystal/Ceramic Resonator** (board có thạch anh 25 MHz) |
| **Clock** | SYSCLK = **400 MHz** (Power → VOS1; PLL từ HSE 25MHz, cấu hình ST-validated cho build báo cáo). |
| **Cortex-M7** | bật **I-Cache** + **D-Cache**. CPU_CACHE_Enable() do CubeMX gen. |
| **SDMMC1** | mode **SD 4 bits Wide bus**. Pins: PC8–PC11 (D0–D3), PC12 (CK), PD2 (CMD). |
| **FATFS** | bật **SD Card**. Cấu hình ffconf (§3). |
| **QUADSPI** | Bank1. Pins: **PB2** CLK, **PB6** BK1_NCS, **PD11/PD12** IO0/IO1, **PE2** IO2, **PD13** IO3. FlashSize = **22** (2^23 = 8MB), ChipSelectHighTime ≥ 1, ClockPrescaler 2–3 lúc bring-up. |
| **USART1** | **Asynchronous**, 115200 8N1. Pins **PA9** (TX), **PA10** (RX). Handle phải tên `huart1`. |
| **MPU** | có thể để CubeMX không cấu hình QSPI — `board.c` tự set vùng `0x90000000` 8MB **non-cacheable**. (Nếu CubeMX có MPU_Config riêng, vẫn OK vì `board_init()` ghi đè region 0.) |
| **DWT** | không cần CubeMX; `board.c` bật trong `board_init()`. |

⚠ **Đối chiếu schematic** WeAct MiniSTM32H7xx cho QSPI IO2=PE2 / IO3=PD13 trước khi generate
(các pin còn lại là chuẩn). Schematic: github.com/WeActStudio/MiniSTM32H7xx → HARDWARE/.

---

## 3. FatFS `ffconf.h` (qua CubeMX FATFS → Advanced/Parameters)

| Tham số | Giá trị | Lý do |
|---|---|---|
| `FF_USE_STRFUNC` | **1** | dùng `f_gets` (đọc manifest.csv từng dòng) |
| `FF_USE_LFN`     | **1, 2 hoặc 3** | tên file dài (`test_data.pack`, `*.tflite`) |
| `FF_FS_RPATH`    | ≥ 0 | (không bắt buộc) |
| `FF_FS_READONLY` | 0 | cần ghi `results.csv` + `f_mkdir` |
| `FF_MAX_SS`      | 512 (hoặc 4096 nếu thẻ format cluster lớn) | |

(LFN=2/3 cần heap/stack cho buffer; LFN=1 dùng static — chọn theo RAM.)

---

## 4. Thêm code harness vào project

1. Copy thư mục `stm32/app/*.c,*.h` vào project (vd `Core/App/`).
2. CubeIDE → project properties → **Include paths** thêm thư mục đó.
3. Thêm các `.c` vào build (CubeIDE tự nhận nếu trong Source folder).
4. Trong **`Core/Src/main.c`**, gọi `kernel_main()` sau khi peripheral init xong:

```c
/* USER CODE BEGIN Includes */
#include "kernel.h"
/* USER CODE END Includes */

int main(void) {
  /* ... HAL_Init(); SystemClock_Config(); MX_*_Init() do CubeMX gen ... */
  /* (CubeMX đã gọi MPU_Config()/CPU_CACHE_Enable() nếu bật cache) */

  /* USER CODE BEGIN 2 */
  kernel_main();     /* không return */
  /* USER CODE END 2 */

  while (1) { }
}
```

> Handle CubeMX phải đúng tên mặc định: `huart1`, `hqspi`, `hsd1` + FatFS (`SDFatFS`, `SDPath`).
> Nếu CubeMX đặt tên khác (vd `hsd`), sửa `extern` ở `sdcard.c`/`board.c`/`qspi_model.c`.

**Compiler:** `-Os`, `-mfpu=fpv5-d16 -mfloat-abi=hard`. **Không** cần `-u _printf_float`
(harness format số bằng integer).

---

## 5. Build & flash (CubeIDE-for-VSCode + ST-Link đã cắm)

- Build trong extension (hoặc `make`/CMake do CubeMX gen).
- Kiểm tra **map file**: `.bss` ~ 49KB (`s_input` 48KB + buffer khác). Phải vừa RAM.
  Nếu tràn → cảnh báo, cân nhắc đặt buffer vào AXI SRAM (`__attribute__((section(".RAM_D1")))`).
- Flash qua ST-Link. Mở serial monitor **115200 8N1** trên cổng VCP (USART1/PA9-PA10 qua
  USB-UART; nếu board không có bridge, nối USB-TTL vào PA9/PA10).

---

## 6. Milestone (mỗi mốc xác minh bằng log UART, KHÔNG cần weight)

| Mốc | Kỳ vọng trên UART |
|---|---|
| **M1 Init** | banner `[board]…`, `[sd] mounted '0:/' total=… free=…`, `[qspi] JEDEC ID = EF 40 17` |
| **M2 Manifest+QSPI** | mỗi model: `[qspi] erase… program… loaded OK, CRC=…` (CRC khớp manifest). Nếu `CRC MISMATCH` → lỗi nạp/đọc QSPI. |
| **M3 Data path** | `[pack] N=519 3x64x64 stride=49156`, chạy hết 519 sample không `bad label`. |
| **M4 Metrics** | dòng `[kernel] DONE <id> acc=…/10000 f1=…/10000 lat_avg=… us p95=… us N=519`; file `/results/results.csv` có dòng tương ứng. |
| **M5 End-to-end** | lặp hết các model trong manifest, kết `=== xong: N model -> results/results.csv ===`. |

> **STUB inference**: accuracy ở Phase 3 **không có ý nghĩa** (logits giả), chỉ để xác minh
> luồng + latency là số thật. Accuracy thật đến ở Phase 4 (TFLM).

---

## 7. JEDEC ID tham chiếu (W25Q64)
`EF 40 17` = Winbond / type 0x40 / capacity 0x17 (8MB). Khác → kiểm tra lại QSPI pins/prescaler.

---

## 8. Sang Phase 4 (sau, cần weight + TF)
- Thay `inference.c` STUB bằng **TFLite-Micro + CMSIS-NN**, đọc model tại `ctx->slot_base`.
- Convert 12 model GO `.pth → .tflite INT8` (cần cài TensorFlow + onnx2tf/ai-edge-torch).
- Khi binary > 128KB internal flash → chuyển app sang **QSPI XIP** + routine nạp QSPI chạy từ
  **RAM**. Xem [PLAN.md](PLAN.md) §8. Memory map slot/app trong `model_ctx.h` đã chừa sẵn 2MB đầu QSPI.
