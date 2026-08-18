/* model_ctx.h — hợp đồng kernel ↔ app (1 ảnh firmware, ranh giới module sạch).
 *
 * Kernel điền struct này cho từng model trong manifest; các module app
 * (test_pack, inference, metrics) chỉ đọc. Xem stm32/PLAN.md §7.4.
 */
#ifndef MODEL_CTX_H
#define MODEL_CTX_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define JUJUBE_NUM_CLASSES   6
#define JUJUBE_INPUT_W       64
#define JUJUBE_INPUT_H       64
#define JUJUBE_INPUT_C       3
#define JUJUBE_INPUT_FLOATS  (JUJUBE_INPUT_C * JUJUBE_INPUT_H * JUJUBE_INPUT_W) /* 12288 */

/* QSPI memory-map (W25Q64, 8MB @ 0x90000000). Giữ 2MB đầu cho app XIP tương lai
 * (Phase 4); model slot 6MB ngay sau. Xem PLAN.md §7.3. */
#define QSPI_BASE_ADDR       0x90000000u
#define MODEL_SLOT_OFFSET    0x00200000u            /* 2 MB */
#define MODEL_SLOT_BASE      (QSPI_BASE_ADDR + MODEL_SLOT_OFFSET)
#define MODEL_SLOT_SIZE      0x00600000u            /* 6 MB */

#define MODEL_ID_MAX         48
#define PATH_MAX_LEN         96

typedef struct {
    char     model_id[MODEL_ID_MAX];   /* tên model (= key results.csv) */
    char     model_path[PATH_MAX_LEN]; /* đường dẫn .tflite/.bin trên SD */
    uint32_t input_w, input_h, input_c;
    uint32_t num_classes;
    uint32_t crc32_expected;           /* CRC32 từ manifest để verify QSPI */

    uint32_t slot_base;                /* địa chỉ memory-mapped của model trong QSPI */
    uint32_t model_size;               /* số byte model đã nạp vào slot */

    /* buffer SRAM dùng chung (kernel cấp phát static, app mượn) */
    float   *input_buf;                /* >= JUJUBE_INPUT_FLOATS phần tử */
    float   *output_buf;               /* >= num_classes phần tử (logits) */
} model_ctx_t;

#ifdef __cplusplus
}
#endif

#endif /* MODEL_CTX_H */
