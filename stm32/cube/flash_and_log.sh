#!/usr/bin/env bash
# Flash jujube_h750.elf qua ST-Link, chạy, rồi DUMP RAM log mailbox (g_logbuf) qua
# ST-Link — đọc log mà KHÔNG cần UART/COM. Tham số 1 (tùy chọn): số giây chờ chạy (mặc định 4).
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
B="/c/Users/DELL/AppData/Local/stm32cube/bundles"
PRG="$B/programmer/2.22.0+st.1/bin/STM32_Programmer_CLI.exe"
NM="$B/gnu-tools-for-stm32/14.3.1+st.2/bin/arm-none-eabi-nm.exe"
ELF="$HERE/build/jujube_h750.elf"
# XIP build tạo _flash.hex (không có QSPI gap) — dùng hex đó để flash
FLASH_IMG="$HERE/build/jujube_h750_flash.hex"
[ -f "$FLASH_IMG" ] || FLASH_IMG="$ELF"
WAIT="${1:-4}"

echo "=== flash ($FLASH_IMG) ==="
"$PRG" -c port=SWD mode=UR -w "$FLASH_IMG" -rst 2>&1 | grep -iE 'verified|error|reset'

LOGBUF=$("$NM" "$ELF" | awk '/ g_logbuf$/{print "0x"$1}')
echo "=== chạy ${WAIT}s rồi đọc log @ $LOGBUF ==="
sleep "$WAIT"
"$PRG" -c port=SWD mode=HOTPLUG -r "$LOGBUF" 0x8000 /tmp/jujube_log.bin >/dev/null 2>&1
echo "----------------------------------------"
tr -d '\000' < /tmp/jujube_log.bin
echo "----------------------------------------"
