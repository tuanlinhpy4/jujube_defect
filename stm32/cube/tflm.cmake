# tflm.cmake — TFLite-Micro + CMSIS-NN cho jujube_h750 (Phase 4).
#
# Nguồn lấy từ cây clone .tflm_clone/ (gitignore) — xem PLAN.md §P4.
#   - TFLM: tflite-micro @074b75f8 (clone --depth 1)
#   - deps pinned (downloads/): flatbuffers v25.9.23, gemmlowp 719139ce,
#     ruy d3712831, CMSIS-NN 6d9d61d8  (khớp version TFLM ghim)
#   - CMSIS-NN dùng API s8 hiện đại (KHÔNG dùng CMSIS-NN q7 cũ trong pack H7).
#
# Định nghĩa: TFLM_SRCS, TFLM_INCLUDE_DIRS, TFLM_DEFINES (dùng ở CMakeLists.txt).

set(TFLM_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.tflm_clone")
set(TFLM_TF   "${TFLM_ROOT}/tensorflow")
set(TFLM_DL   "${TFLM_TF}/lite/micro/tools/make/downloads")
set(CMSIS_NN  "${TFLM_DL}/cmsis_nn")

if(NOT EXISTS "${TFLM_TF}/lite/micro/micro_interpreter.cc")
    message(FATAL_ERROR "TFLM source not found at ${TFLM_TF} — chạy bước tải nguồn (PLAN.md P4).")
endif()

# --- 1. TFLM core (micro base) ---
file(GLOB TFLM_MICRO
    "${TFLM_TF}/lite/micro/*.cc"
    "${TFLM_TF}/lite/micro/arena_allocator/*.cc"
    "${TFLM_TF}/lite/micro/memory_planner/*.cc"
    "${TFLM_TF}/lite/micro/tflite_bridge/*.cc")

# --- 2. Shared TfLite (TFL_CC_SRCS = mọi .cc/.c trong tensorflow/ trừ micro & experimental) ---
file(GLOB_RECURSE TFLM_SHARED "${TFLM_TF}/*.cc" "${TFLM_TF}/*.c")
list(FILTER TFLM_SHARED EXCLUDE REGEX "/lite/micro/")
list(FILTER TFLM_SHARED EXCLUDE REGEX "/lite/experimental/")
list(FILTER TFLM_SHARED EXCLUDE REGEX "/array\\.cc$")          # cần TF_LITE_STATIC_MEMORY -> loại array.cc

# --- 3. Kernels: bản reference, THAY các op có bản cmsis_nn (giữ *_common.cc) ---
file(GLOB TFLM_KERNELS       "${TFLM_TF}/lite/micro/kernels/*.cc")
file(GLOB TFLM_CMSIS_KERNELS "${TFLM_TF}/lite/micro/kernels/cmsis_nn/*.cc")
foreach(cf ${TFLM_CMSIS_KERNELS})
    get_filename_component(cb "${cf}" NAME)
    list(FILTER TFLM_KERNELS EXCLUDE REGEX "/kernels/${cb}$")  # loại reference cùng tên (add/conv/...)
endforeach()

# --- 4. CMSIS-NN C source (Cortex-M7 -> tự bật DSP qua __ARM_FEATURE_DSP) ---
file(GLOB_RECURSE TFLM_CMSIS_NN_SRC "${CMSIS_NN}/Source/*.c")
list(FILTER TFLM_CMSIS_NN_SRC EXCLUDE REGEX "/Bindings/")

set(TFLM_SRCS
    ${TFLM_MICRO} ${TFLM_SHARED} ${TFLM_KERNELS} ${TFLM_CMSIS_KERNELS}
    ${TFLM_CMSIS_NN_SRC})

# --- 5. Loại file test/helper/không cần (tránh kéo retarget/NPU) ---
list(FILTER TFLM_SRCS EXCLUDE REGEX "_test\\.cc$")
list(FILTER TFLM_SRCS EXCLUDE REGEX "test_helpers?\\.cc$")
list(FILTER TFLM_SRCS EXCLUDE REGEX "/fake_micro_context\\.cc$")
list(FILTER TFLM_SRCS EXCLUDE REGEX "/mock_")
list(FILTER TFLM_SRCS EXCLUDE REGEX "/kernel_runner\\.cc$")     # chỉ dùng cho unit test kernel
list(FILTER TFLM_SRCS EXCLUDE REGEX "/debug_log\\.cc$")         # ta tự cấp DebugLog -> UART (tflm_shim.cc)
list(FILTER TFLM_SRCS EXCLUDE REGEX "/ethosu\\.cc$")            # Ethos-U NPU, không có trên H750

set(TFLM_INCLUDE_DIRS
    ${TFLM_ROOT}
    ${TFLM_DL}/flatbuffers/include
    ${TFLM_DL}/gemmlowp
    ${TFLM_DL}/ruy
    ${CMSIS_NN}            # TFLM cmsis_nn kernel include dạng "Include/arm_nnfunctions.h"
    ${CMSIS_NN}/Include)   # source CMSIS-NN include dạng "arm_nnfunctions.h"

# TF_LITE_STRIP_ERROR_STRINGS: bỏ toàn bộ chuỗi báo lỗi/log của TFLM (MicroPrintf) khỏi
#   .rodata — tiết kiệm ~10KB flash, đủ để build 13-op (đủ op cho junet_edge_l + squeezenet1_1)
#   vừa 128KB internal flash, KHỎI cần QSPI XIP. Log của ta (uart_log) không bị ảnh hưởng.
set(TFLM_DEFINES TF_LITE_STATIC_MEMORY TF_LITE_MCU_DEBUG_LOG TF_LITE_STRIP_ERROR_STRINGS CMSIS_NN)

list(LENGTH TFLM_SRCS _tflm_n)
message(STATUS "TFLM: ${_tflm_n} source files (TFLM core + kernels + CMSIS-NN)")
