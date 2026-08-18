/* test_pack.h — reader test_data.pack, stream từng sample vào SRAM. PACK_FORMAT.md §1. */
#ifndef TEST_PACK_H
#define TEST_PACK_H

#include <stdint.h>

/* Mở pack + validate header. Trả 0 OK, !=0 lỗi. */
int      test_pack_open(const char *path);
uint32_t test_pack_count(void);       /* số sample (N) */
uint32_t test_pack_floats(void);      /* C*H*W mỗi sample */

/* Đọc sample idx: img_out[C*H*W] (float32 NCHW), *label_out (0..C-1). Trả 0 OK. */
int      test_pack_read(uint32_t idx, float *img_out, uint32_t *label_out);
void     test_pack_close(void);

#endif /* TEST_PACK_H */
