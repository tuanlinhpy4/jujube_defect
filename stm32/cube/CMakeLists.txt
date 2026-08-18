cmake_minimum_required(VERSION 3.22)

# Toolchain phải set trước project()
set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/cmake/gcc-arm-none-eabi.cmake")

project(jujube_h750 C CXX ASM)

# --- STM32Cube FW pack (HAL + CMSIS + FatFs) đã cài sẵn ---
set(PACK "C:/Users/DELL/STM32Cube/Repository/STM32Cube_FW_H7_V1.12.1")
set(HAL  "${PACK}/Drivers/STM32H7xx_HAL_Driver")
set(CMSIS "${PACK}/Drivers/CMSIS")
set(FATFS "${PACK}/Middlewares/Third_Party/FatFs/src")
set(APP   "${CMAKE_CURRENT_SOURCE_DIR}/../app")

set(MCU_FLAGS -mcpu=cortex-m7 -mthumb -mfpu=fpv5-d16 -mfloat-abi=hard)
add_compile_definitions(USE_HAL_DRIVER STM32H750xx)

# HAL: gom hal_*.c (module không bật sẽ rỗng nhờ #ifdef MODULE_ENABLED), loại *_template.c,
# thêm core stm32h7xx_hal.c (HAL_Init/Tick — không khớp pattern hal_*) + ll_sdmmc.
file(GLOB HAL_SRCS "${HAL}/Src/stm32h7xx_hal_*.c")
list(FILTER HAL_SRCS EXCLUDE REGEX ".*_template\\.c$")
list(APPEND HAL_SRCS "${HAL}/Src/stm32h7xx_hal.c" "${HAL}/Src/stm32h7xx_ll_sdmmc.c")

# FatFs: chỉ ff.c + unicode (LFN). diskio do ta tự viết (Core/Src/sd_diskio.c),
# KHÔNG dùng pack diskio.c/ff_gen_drv.c.
set(FATFS_SRCS "${FATFS}/ff.c" "${FATFS}/option/unicode.c")

set(APP_SRCS
    ${APP}/board.c ${APP}/sdcard.c ${APP}/qspi_model.c ${APP}/manifest.c
    ${APP}/test_pack.c ${APP}/metrics.c ${APP}/results.c
    ${APP}/crc32.c ${APP}/kernel.c)

set(TFLM_APP_SRCS ${APP}/inference.cc ${APP}/tflm_shim.cc)   # backend TFLM (Phase 4)

# --- TFLite-Micro + CMSIS-NN (định nghĩa TFLM_SRCS / TFLM_INCLUDE_DIRS / TFLM_DEFINES) ---
include(${CMAKE_CURRENT_SOURCE_DIR}/tflm.cmake)

set(TFLM_COMPILE_OPTIONS
    ${MCU_FLAGS}
    -ffunction-sections -fdata-sections
    -fno-unwind-tables -fno-asynchronous-unwind-tables
    $<$<CONFIG:Debug>:-Og -g3>
    $<$<NOT:$<CONFIG:Debug>>:-Os -g>
    $<$<COMPILE_LANGUAGE:CXX>:-std=gnu++17>
    $<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions>
    $<$<COMPILE_LANGUAGE:CXX>:-fno-rtti>
    $<$<COMPILE_LANGUAGE:CXX>:-fno-threadsafe-statics>
    $<$<COMPILE_LANGUAGE:CXX>:-fno-use-cxa-atexit>
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-parameter>
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-sign-compare>)

# Bước A: chỉ 8 op của mnasnet -> binary nhỏ để thử vừa 128KB internal flash (kiểm đúng
# data-path trước khi làm XIP). Union 13-op tràn flash -> cần QSPI XIP (Bước B).
option(MNASNET_ONLY "Chỉ đăng ký 8 op mnasnet (test Bước A)" OFF)
option(JUJUBE_RAM_ONLY "Chỉ đo TFLM arena RAM sau AllocateTensors, không chạy test pack" OFF)

# Bước B: XIP build — TFLM code chạy từ QSPI 0x90000000
option(XIP_BUILD "Đặt TFLM vào QSPI XIP (Bước B)" OFF)
option(JUJUBE_SYSCLK_480 "Run STM32H750 at 480 MHz using VOS0" OFF)
if(XIP_BUILD)
    set(LDSCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/STM32H750VBTx_XIP.ld")
    # Build TFLM + inference backend thành static library riêng
    # → linker script match bằng *libtflm_xip.a:* để đặt vào QSPI_APP
    add_library(tflm_xip STATIC ${TFLM_SRCS} ${TFLM_APP_SRCS})
    target_include_directories(tflm_xip PRIVATE
        Core/Inc ${APP}
        ${HAL}/Inc
        ${CMSIS}/Device/ST/STM32H7xx/Include
        ${CMSIS}/Include
        ${TFLM_INCLUDE_DIRS})
    target_compile_definitions(tflm_xip PRIVATE USE_HAL_DRIVER STM32H750xx ${TFLM_DEFINES})
    if(MNASNET_ONLY)
        target_compile_definitions(tflm_xip PRIVATE TFLM_OPS_MNASNET)
    endif()
    target_compile_options(tflm_xip PRIVATE ${TFLM_COMPILE_OPTIONS})
else()
    set(LDSCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/STM32H750VBTx_FLASH.ld")
    if(MNASNET_ONLY)
        set_property(SOURCE ${APP}/inference.cc APPEND PROPERTY COMPILE_DEFINITIONS TFLM_OPS_MNASNET)
    endif()
endif()

set(CORE_SRCS
    Core/Src/main.c
    Core/Src/stm32h7xx_hal_msp.c
    Core/Src/stm32h7xx_it.c
    Core/Src/system_stm32h7xx.c
    Core/Src/sd_diskio.c)

if(XIP_BUILD)
    add_executable(${PROJECT_NAME}.elf
        ${CORE_SRCS} ${HAL_SRCS} ${FATFS_SRCS} ${APP_SRCS}
        startup_stm32h750xx.s)
    target_link_libraries(${PROJECT_NAME}.elf PRIVATE tflm_xip)
else()
    add_executable(${PROJECT_NAME}.elf
        ${CORE_SRCS} ${HAL_SRCS} ${FATFS_SRCS} ${APP_SRCS} ${TFLM_APP_SRCS} ${TFLM_SRCS}
        startup_stm32h750xx.s)
endif()

target_include_directories(${PROJECT_NAME}.elf PRIVATE
    Core/Inc
    ${APP}
    ${HAL}/Inc
    ${CMSIS}/Device/ST/STM32H7xx/Include
    ${CMSIS}/Include
    ${FATFS}
    ${TFLM_INCLUDE_DIRS})

target_compile_definitions(${PROJECT_NAME}.elf PRIVATE ${TFLM_DEFINES})
if(JUJUBE_RAM_ONLY)
    target_compile_definitions(${PROJECT_NAME}.elf PRIVATE JUJUBE_RAM_ONLY)
endif()
if(JUJUBE_SYSCLK_480)
    target_compile_definitions(${PROJECT_NAME}.elf PRIVATE JUJUBE_SYSCLK_480)
endif()

target_compile_options(${PROJECT_NAME}.elf PRIVATE
    ${MCU_FLAGS}
    -ffunction-sections -fdata-sections
    -fno-unwind-tables -fno-asynchronous-unwind-tables
    $<$<CONFIG:Debug>:-Og -g3>
    $<$<NOT:$<CONFIG:Debug>>:-Os -g>
    # cảnh báo chặt cho code app/HAL (C); TFLM ồn -> hạ bớt riêng cho C++
    $<$<COMPILE_LANGUAGE:C>:-Wall>
    $<$<COMPILE_LANGUAGE:C>:-Wextra>
    $<$<COMPILE_LANGUAGE:C>:-Wno-unused-parameter>
    $<$<COMPILE_LANGUAGE:CXX>:-std=gnu++17>
    $<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions>
    $<$<COMPILE_LANGUAGE:CXX>:-fno-rtti>
    $<$<COMPILE_LANGUAGE:CXX>:-fno-threadsafe-statics>
    $<$<COMPILE_LANGUAGE:CXX>:-fno-use-cxa-atexit>
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-parameter>
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-sign-compare>)

target_link_options(${PROJECT_NAME}.elf PRIVATE
    ${MCU_FLAGS}
    -T${LDSCRIPT}
    --specs=nano.specs --specs=nosys.specs
    -Wl,--gc-sections
    -Wl,--print-memory-usage
    -Wl,-Map=${PROJECT_NAME}.map,--cref
    -lc -lm)

# .hex + .bin + in size
if(XIP_BUILD)
    # Flash hex: chỉ internal-flash sections (không có QSPI gap → file nhỏ, dùng với ST-Link)
    # tflm_app.bin: chỉ QSPI sections → copy lên SD card tên "tflm_app.bin"
    add_custom_command(TARGET ${PROJECT_NAME}.elf POST_BUILD
        COMMAND ${CMAKE_OBJCOPY} -O ihex
            --remove-section=.qspi_text --remove-section=.qspi_rodata
            $<TARGET_FILE:${PROJECT_NAME}.elf> ${PROJECT_NAME}_flash.hex
        COMMAND ${CMAKE_OBJCOPY} -O binary
            --only-section=.qspi_text --only-section=.qspi_rodata
            $<TARGET_FILE:${PROJECT_NAME}.elf> tflm_app.bin
        COMMAND ${CMAKE_SIZE} $<TARGET_FILE:${PROJECT_NAME}.elf>)
else()
    add_custom_command(TARGET ${PROJECT_NAME}.elf POST_BUILD
        COMMAND ${CMAKE_OBJCOPY} -O ihex   $<TARGET_FILE:${PROJECT_NAME}.elf> ${PROJECT_NAME}.hex
        COMMAND ${CMAKE_OBJCOPY} -O binary $<TARGET_FILE:${PROJECT_NAME}.elf> ${PROJECT_NAME}.bin
        COMMAND ${CMAKE_SIZE} $<TARGET_FILE:${PROJECT_NAME}.elf>)
endif()
