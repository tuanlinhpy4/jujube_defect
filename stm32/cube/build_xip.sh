#!/usr/bin/env bash
# Build XIP firmware (Bước B) — TFLM code chạy từ QSPI 0x90000000.
# Trước khi flash cần chạy: python stm32/host/flash_tflm_to_qspi.py build/jujube_h750.elf
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
B="/c/Users/DELL/AppData/Local/stm32cube/bundles"
export PATH="$B/cmake/4.3.1+st.1/bin:$B/ninja/1.13.2+st.1/bin:$PATH"
NINJA="$B/ninja/1.13.2+st.1/bin/ninja.exe"

echo "=== Build XIP (TFLM @ QSPI, 13-op full) ==="

# Force XIP_BUILD=ON + MNASNET_ONLY=OFF
cmake.exe -G Ninja -DCMAKE_MAKE_PROGRAM="$NINJA" -DCMAKE_BUILD_TYPE=Release \
    -DXIP_BUILD=ON -DMNASNET_ONLY=OFF \
    -S "$HERE" -B "$HERE/build"

"$NINJA" -C "$HERE/build"

echo ""
echo "=> $HERE/build/jujube_h750_flash.hex  (nạp vào board qua ST-Link)"
echo "=> $HERE/build/tflm_app.bin           (copy lên SD card, đặt tên tflm_app.bin)"
echo ""
echo "Tiếp theo:"
echo "  1. Copy build/tflm_app.bin → thư mục gốc SD card (tên: tflm_app.bin)"
echo "  2. bash stm32/cube/flash_and_log.sh 30   (flash firmware + xem log)"
echo ""
echo "Lần đầu boot: firmware tự erase+program QSPI từ SD, sau đó XIP bình thường."
