#!/usr/bin/env bash
# Build firmware jujube_h750 bằng toolchain bundle của STM32 VSCode extension.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
B="/c/Users/DELL/AppData/Local/stm32cube/bundles"
export PATH="$B/cmake/4.3.1+st.1/bin:$B/ninja/1.13.2+st.1/bin:$PATH"
NINJA="$B/ninja/1.13.2+st.1/bin/ninja.exe"

cmake.exe -G Ninja -DCMAKE_MAKE_PROGRAM="$NINJA" -DCMAKE_BUILD_TYPE=Release \
    -S "$HERE" -B "$HERE/build"
"$NINJA" -C "$HERE/build"
echo "=> $HERE/build/jujube_h750.{elf,bin,hex}"
