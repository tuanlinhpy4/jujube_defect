# jujube_h750 — project firmware (build tay, không cần CubeMX)

Project STM32CubeMX-free, dựng tay từ firmware pack **STM32Cube_FW_H7 V1.12.1** đã cài,
build bằng toolchain bundle của extension **STM32 for VSCode** (arm-gcc 14.3.1 + CMake + ninja).
Lý do không dùng CubeMX: máy offline → CubeMX headless kẹt ở bước auth/login (xem PLAN.md note).

Board: **WeAct MiniSTM32H750VBT6**, HSE 25MHz → **SYSCLK 400MHz** (VOS1, cấu hình ST-validated).

## Cấu trúc
```
cube/
  CMakeLists.txt              # tham chiếu HAL/CMSIS/FatFs trong pack + ../app + Core
  cmake/gcc-arm-none-eabi.cmake
  STM32H750VBTx_FLASH.ld      # internal flash 128K; .data/.bss ở RAM_D1 (AXI, DMA-capable)
  startup_stm32h750xx.s       # copy từ pack
  Core/Inc/  main.h, stm32h7xx_hal_conf.h (HSE=25MHz), stm32h7xx_it.h, ffconf.h (FatFs R0.12c, LFN)
  Core/Src/  main.c (clock+MX init), stm32h7xx_hal_msp.c (pin WeAct), stm32h7xx_it.c,
             system_stm32h7xx.c, sd_diskio.c (HAL_SD + cache maintenance)
  build.sh                    # cmake + ninja
  flash_and_log.sh            # flash + DUMP log qua ST-Link (không cần UART)
```

## Build
```bash
bash stm32/cube/build.sh
# => build/jujube_h750.elf (.bin .hex). FLASH ~37KB/128KB, RAM ~70KB/512KB.
```

Nếu máy Windows không có `bash`, dùng PowerShell:
```powershell
powershell -ExecutionPolicy Bypass -File stm32/cube/build.ps1
```

## Flash + đọc log (KHÔNG cần UART/COM)
ST-Link V2 trên máy này không có VCP nên không có cổng COM. Firmware mirror toàn bộ `uart_log`
vào **RAM mailbox** `g_logbuf` (magic "JLOG" @ `g_logpos`). Đọc qua ST-Link:
```bash
bash stm32/cube/flash_and_log.sh 4      # flash, chạy 4s, in log từ RAM
```
PowerShell:
```powershell
powershell -ExecutionPolicy Bypass -File stm32/cube/flash_and_log.ps1 -WaitSeconds 4
```
(Nếu có USB-TTL: nối PA9=TX, PA10=RX, GND → xem log trực tiếp 115200 8N1 — giống hệt.)

## P4 — TFLM+CMSIS-NN (2026-06-21, ĐANG LÀM)
`inference.c` STUB đã thay bằng `inference.cc` (TFLite-Micro + CMSIS-NN). Build C++ qua `cube/tflm.cmake`
(nguồn ở `../.tflm_clone`, gitignore). **Compile sạch.** Kích thước (-Os):
- HỢP 13-op (cả 50 model): **144 712 B / 128 KB = 110% → TRÀN** ⇒ cần **QSPI XIP** (xem PLAN.md §8).
- `-DMNASNET_ONLY=ON` (8 op): **126 408 B / 128 KB = 96.4% → vừa** → flash test Bước A:
  ```bash
  cmake -G Ninja -DMNASNET_ONLY=ON -S cube -B cube/build && ninja -C cube/build
  bash cube/flash_and_log.sh 30   # cần thẻ SD có manifest_mnasnet.csv + mnasnet0_5_s*.tflite
  ```
Chi tiết & việc còn lại (XIP, golden 50 model, RELU_0_TO_1): **PLAN.md §8**.

## Trạng thái verify trên phần cứng (đọc qua ST-Link) — ✅ M1–M5 XONG (2026-06-21)
Toàn bộ harness chạy end-to-end với thẻ SD thật + 2 model dummy. Log thực tế:
```
[board] SYSCLK=400000000 Hz, DWT on, MPU QSPI non-cacheable
[sd] mounted '0:/'  total=500 MB  free=473 MB
[qspi] JEDEC ID = EF 40 17
[kernel] DONE dummy_a  acc=2909/10000 f1=2872/10000 lat_avg=1141us p95=1143us N=519  (CRC 162B3EDD)
[kernel] DONE dummy_b  ...                                                   (CRC 9B1EE9AD)
[kernel] === xong: 2 model -> results/results.csv ===
```
(acc/f1/lat chỉ là số STUB — Phase 4 thay TFLM mới có ý nghĩa; 2 model ra giống nhau vì STUB chỉ phụ thuộc input.)

### 3 bug bring-up đã sửa (đừng tái phạm)
1. **`main.c` SystemClock_Config treo ở VOSRDY** → thiếu `HAL_PWREx_ConfigSupply(PWR_LDO_SUPPLY)`
   TRƯỚC khi set VOS (H7 bắt buộc; WeAct dùng LDO). Triệu chứng: PC kẹt `main+0xc6`, RCC ở HSI.
2. **`board.c` log RAM mailbox rỗng qua ST-Link** → D-cache giữ `g_logbuf`, ST-Link đọc SRAM vật lý ra 0.
   Fix: `SCB_CleanDCache_by_Addr` sau mỗi `uart_log` (và sau khi set `g_logmagic`).
3. **`sd_diskio.c` `f_mount` fr=13 (FR_NO_FILESYSTEM)** → `HAL_SD_ReadBlocks` bản polling của H7 truyền
   bằng **CPU FIFO** (không IDMA), nhưng code lại `SCB_InvalidateDCache_by_Addr(s_scratch)` sau read →
   xóa mất data CPU vừa ghi (chưa write-back) → đọc rác → check_fs fail. Fix: BỎ invalidate/clean
   (CPU FIFO đã coherent). ⚠ Nếu sau này đổi sang `HAL_SD_ReadBlocks_DMA` thì PHẢI thêm lại cache op.

### Lưu ý thẻ SD
- Thẻ 128GB SDXC, MBR, partition FAT32 500MB @offset 8MB (LBA 16384) — FatFs R0.12c tự hop MBR OK.
- `_FS_EXFAT=0` trong `ffconf.h` → thẻ PHẢI là FAT32 (không exFAT).
- `flash_and_log.sh 60`: benchmark 2 model ~ chạy trong vài chục giây, chờ 60s là đủ.
