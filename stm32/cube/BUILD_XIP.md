# QSPI XIP Build (Bước B)

## Mục tiêu

Đưa TFLM code (`.text` + `.rodata`) lên QSPI flash để chạy XIP từ `0x90000000`, giải phóng internal flash 128KB cho bootstrap + HAL.

**Lợi ích:**
- 13-op full build vừa (chỉ dùng ~40KB internal flash cho bootstrap)
- Có thể thêm op, thêm model mà không lo tràn flash
- Latency tương đương (QSPI cacheable write-through)

---

## Kiến trúc

```
Internal Flash 128KB (0x08000000):
  ├── Vector table + startup
  ├── HAL (QSPI, UART, SD, clock, ...)
  ├── FatFS
  ├── app: board.c, kernel.c, qspi_model.c, sdcard.c, ...
  └── ~40KB total

QSPI 0x90000000–0x901FFFFF (2MB app slot):
  └── TFLM code (.qspi_text + .qspi_rodata) ~100KB
      → CPU fetch instruction từ đây khi inference

QSPI 0x90200000–0x907FFFFF (6MB model slot):
  └── .tflite model hiện tại (không đổi)
```

**Khi swap model**: `qspi_load_model()` gọi `HAL_QSPI_Abort` → CPU đang fetch từ internal flash → an toàn, không cần `.RamFunc`.

---

## Build & Flash

### Bước 1: Build XIP firmware

```bash
cd stm32/cube
bash build_xip.sh
```

Tạo `build/jujube_h750.elf` với:
- Bootstrap + HAL + app kernel → internal flash
- TFLM + inference.cc → QSPI XIP (LMA `0x90000000`)

### Bước 2: Nạp TFLM code lên QSPI

```bash
python ../host/flash_tflm_to_qspi.py build/jujube_h750.elf
```

Script tự động:
1. Extract `.qspi_text` + `.qspi_rodata` từ ELF → `tflm_code.bin`
2. Nạp vào QSPI offset `0x0` (địa chỉ `0x90000000`) qua ST-Link
3. Reset board

### Bước 3: Flash firmware như bình thường

```bash
bash flash_and_log.sh 30
```

Firmware boot → `qspi_enable_memmap()` → CPU fetch TFLM code từ QSPI XIP → chạy bình thường.

---

## Verify

Log expect:
```
[board] STM32H750 jujube benchmark harness
[board] SYSCLK=400000000 Hz, DWT on, MPU QSPI cacheable (WT)
[qspi] JEDEC ID = EF 40 17
[infer] junet_edge_l_s1 arena_used=311/504 KB  in[type=1 dims=4:64x64x3] out[type=1 n=4]
```

Nếu fail → check:
- `tflm_code.bin` có size > 0 không? (không thì chưa build với `-DXIP_BUILD=ON`)
- `qspi_enable_memmap()` được gọi trong `main.c` chưa?
- Linker script đúng là `STM32H750VBTx_XIP.ld` chưa?

---

## So sánh với build cũ

| Build | Flash used | TFLM code | Model swap |
|-------|-----------|-----------|------------|
| Internal (Bước A) | 123 KB / 128 KB = 96% | internal flash | an toàn (không XIP) |
| XIP (Bước B) | ~40 KB / 128 KB = 31% | QSPI XIP | an toàn (qspi_model.c ở internal) |

Bước B tiết kiệm ~80KB internal flash → có thể thêm op, debug symbols, hoặc các module mới.

---

## Rollback về internal flash

```bash
cd stm32/cube
bash build.sh  # không có -DXIP_BUILD
bash flash_and_log.sh 4
```

Firmware chạy toàn bộ từ internal flash như trước, không cần nạp QSPI.
