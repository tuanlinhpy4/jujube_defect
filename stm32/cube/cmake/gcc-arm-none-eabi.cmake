# Toolchain cho STM32H750 (Cortex-M7) dùng arm-none-eabi-gcc bundle của STM32 VSCode ext.
set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR arm)

# Đường dẫn bundle (sửa nếu version đổi).
set(TOOLCHAIN_BIN "C:/Users/DELL/AppData/Local/stm32cube/bundles/gnu-tools-for-stm32/14.3.1+st.2/bin")
set(TOOLCHAIN_PREFIX "${TOOLCHAIN_BIN}/arm-none-eabi-")

set(CMAKE_C_COMPILER   "${TOOLCHAIN_PREFIX}gcc.exe")
set(CMAKE_ASM_COMPILER "${TOOLCHAIN_PREFIX}gcc.exe")
set(CMAKE_CXX_COMPILER "${TOOLCHAIN_PREFIX}g++.exe")
set(CMAKE_OBJCOPY      "${TOOLCHAIN_PREFIX}objcopy.exe" CACHE INTERNAL "")
set(CMAKE_SIZE         "${TOOLCHAIN_PREFIX}size.exe"    CACHE INTERNAL "")

# Không link khi test compiler (bare-metal, không có host runtime).
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

set(CPU_FLAGS "-mcpu=cortex-m7 -mthumb -mfpu=fpv5-d16 -mfloat-abi=hard")
set(CMAKE_C_FLAGS_INIT   "${CPU_FLAGS}")
set(CMAKE_ASM_FLAGS_INIT "${CPU_FLAGS} -x assembler-with-cpp")
set(CMAKE_CXX_FLAGS_INIT "${CPU_FLAGS}")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
