/* qspi_model.h — nạp model SD→QSPI slot (W25Q64), verify CRC32, memory-mapped read.
 * CubeMX lo MX_QUADSPI_Init (FlashSize, prescaler). Driver lệnh W25Q64 ở đây. */
#ifndef QSPI_MODEL_H
#define QSPI_MODEL_H

#include <stdint.h>
#include "ff.h"

/* Sanity: đọc JEDEC ID, log. Trả 0 nếu nhận diện được flash (Winbond W25Q64). */
int qspi_init(void);

/* Nạp file model (SD, path tuyệt đối vd "0:/models/x.tflite") vào slot tại
 * flash-offset 'flash_ofs' (vd MODEL_SLOT_OFFSET). Erase→program→memmap→CRC32.
 * Ghi *out_size = số byte đã nạp. Trả 0 nếu CRC khớp 'crc_expected', !=0 nếu lỗi. */
int qspi_load_model(const char *path, uint32_t flash_ofs,
                    uint32_t crc_expected, uint32_t *out_size);

/* Bật memory-mapped (CPU đọc trực tiếp tại 0x90000000+). */
int qspi_enable_memmap(void);

/* Nạp app XIP từ SD (path, vd "0:/tflm_app.bin") vào QSPI offset 0 nếu slot rỗng.
 * Phải gọi TRƯỚC qspi_enable_memmap(). Bỏ qua nếu QSPI[0..3] != 0xFFFFFFFF. */
int qspi_load_app_if_needed(const char *path);

#endif /* QSPI_MODEL_H */
