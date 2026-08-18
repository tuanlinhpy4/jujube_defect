/* manifest.h — parse /manifest.csv (xem PACK_FORMAT.md §2). */
#ifndef MANIFEST_H
#define MANIFEST_H

#include <stdint.h>
#include "model_ctx.h"

typedef struct {
    char     model_id[MODEL_ID_MAX];
    char     model_bin[PATH_MAX_LEN];
    uint32_t input_w, input_h, input_c;
    uint32_t num_classes;
    uint32_t crc32;
} manifest_entry_t;

/* Mở manifest, bỏ qua dòng header. Trả 0 OK, !=0 lỗi. */
int  manifest_open(const char *path);
/* Lấy entry kế tiếp: 1 = có, 0 = hết file, <0 = lỗi parse. */
int  manifest_next(manifest_entry_t *out);
void manifest_close(void);

#endif /* MANIFEST_H */
