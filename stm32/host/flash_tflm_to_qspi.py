#!/usr/bin/env python3
# flash_tflm_to_qspi.py -- Extract TFLM XIP sections from ELF and flash to QSPI 0x90000000
# Usage: python flash_tflm_to_qspi.py <jujube_h750.elf>
import sys, subprocess, os
from pathlib import Path

QSPI_BASE    = 0x90000000
OBJCOPY      = r"C:\Users\DELL\AppData\Local\stm32cube\bundles\gnu-tools-for-stm32\14.3.1+st.2\bin\arm-none-eabi-objcopy.exe"
PROGRAMMER   = r"C:\Users\DELL\AppData\Local\stm32cube\bundles\programmer\2.22.0+st.1\bin\STM32_Programmer_CLI.exe"

def extract(elf, out):
    subprocess.run([OBJCOPY, "-O", "binary",
                    "--only-section=.qspi_text", "--only-section=.qspi_rodata",
                    elf, out], check=True)
    size = os.path.getsize(out)
    print(f"[extract] {out} ({size} bytes)")
    return size

def flash(bin_path, addr):
    cmd = [PROGRAMMER, "-c", "port=SWD", "mode=UR",
           "-w", bin_path, hex(addr), "-v", "-rst"]
    print(f"[flash] {' '.join(cmd)}")
    r = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    out = (r.stdout or "") + (r.stderr or "")
    for line in out.splitlines():
        if any(k in line for k in ("verified","error","Error","Error","reset","Reset")):
            print(line)
    if r.returncode != 0:
        print(out, file=sys.stderr)
        sys.exit(1)
    print(f"[flash] OK @ {hex(addr)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python flash_tflm_to_qspi.py <jujube_h750.elf>"); sys.exit(1)
    elf = Path(sys.argv[1])
    if not elf.exists():
        print(f"ERROR: {elf} not found"); sys.exit(1)
    out = elf.parent / "tflm_code.bin"
    print("=== Flash TFLM code to QSPI (Buoc B) ===")
    size = extract(str(elf), str(out))
    if size == 0:
        print("ERROR: empty bin -- did you build with -DXIP_BUILD=ON?"); sys.exit(1)
    flash(str(out), QSPI_BASE)
    print("=== Done. Run flash_and_log.sh to flash ELF ===")
