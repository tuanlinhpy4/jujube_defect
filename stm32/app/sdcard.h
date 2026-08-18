/* sdcard.h — mount FatFS trên SDMMC1 + helper đọc file. CubeMX lo SDMMC1 + MX_FATFS_Init. */
#ifndef SDCARD_H
#define SDCARD_H

#include <stdint.h>
#include "ff.h"   /* FatFS (CubeMX middleware) */

/* Mount thẻ SD (logical drive CubeMX). Trả 0 nếu OK, !=0 = FRESULT lỗi. */
int sdcard_mount(void);

/* Mở file đọc-only. Trả FRESULT. */
FRESULT sdcard_open_read(FIL *fp, const char *path);

/* Đọc len byte tại offset 'ofs' vào buf. Trả FRESULT (FR_OK + *br=len nếu đủ). */
FRESULT sdcard_read_at(FIL *fp, uint32_t ofs, void *buf, uint32_t len, UINT *br);

#endif /* SDCARD_H */
