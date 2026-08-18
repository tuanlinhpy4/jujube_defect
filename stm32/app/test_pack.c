/* test_pack.c — đọc test_data.pack (header 40B + N record stride). */
#include "test_pack.h"
#include "sdcard.h"
#include "board.h"
#include "model_ctx.h"

#include <string.h>

#define PACK_MAGIC   "JPK1"
#define PACK_HEADER  40u

static FIL      s_fil;
static int      s_open = 0;
static uint32_t s_n, s_c, s_h, s_w, s_label_bytes, s_stride, s_img_bytes;

int test_pack_open(const char *path) {
    if (s_open) { f_close(&s_fil); s_open = 0; }
    if (sdcard_open_read(&s_fil, path) != FR_OK) return -1;

    uint8_t hdr[PACK_HEADER];
    UINT br;
    if (sdcard_read_at(&s_fil, 0, hdr, PACK_HEADER, &br) != FR_OK || br != PACK_HEADER) {
        f_close(&s_fil); return -2;
    }
    if (memcmp(hdr, PACK_MAGIC, 4) != 0) {
        uart_log("[pack] bad magic\r\n"); f_close(&s_fil); return -3;
    }
    /* header sau magic: [ver,N,C,H,W,dtype,layout,label_bytes,reserved] = u[0..8] */
    uint32_t u[9];
    memcpy(u, hdr + 4, sizeof(u));
    uint32_t version = u[0], dtype = u[5], layout = u[6];
    s_n = u[1]; s_c = u[2]; s_h = u[3]; s_w = u[4];
    s_label_bytes = u[7];

    if (version != 1 || dtype != 0 || layout != 0) {
        uart_log("[pack] unsupported ver=%lu dtype=%lu layout=%lu\r\n",
                 (unsigned long)version, (unsigned long)dtype, (unsigned long)layout);
        f_close(&s_fil); return -4;
    }
    s_img_bytes = s_c * s_h * s_w * 4u;
    s_stride    = s_img_bytes + s_label_bytes;
    s_open = 1;
    uart_log("[pack] N=%lu %lux%lux%lu stride=%lu\r\n",
             (unsigned long)s_n, (unsigned long)s_c, (unsigned long)s_h,
             (unsigned long)s_w, (unsigned long)s_stride);
    return 0;
}

uint32_t test_pack_count(void)  { return s_open ? s_n : 0; }
uint32_t test_pack_floats(void) { return s_open ? s_c * s_h * s_w : 0; }

int test_pack_read(uint32_t idx, float *img_out, uint32_t *label_out) {
    if (!s_open || idx >= s_n) return -1;
    uint32_t base = PACK_HEADER + idx * s_stride;
    UINT br;
    if (sdcard_read_at(&s_fil, base, img_out, s_img_bytes, &br) != FR_OK || br != s_img_bytes)
        return -2;
    uint32_t lbl = 0;
    if (sdcard_read_at(&s_fil, base + s_img_bytes, &lbl, s_label_bytes, &br) != FR_OK)
        return -3;
    if (lbl >= JUJUBE_NUM_CLASSES) { uart_log("[pack] bad label %lu@%lu\r\n",
        (unsigned long)lbl, (unsigned long)idx); return -4; }
    *label_out = lbl;
    return 0;
}

void test_pack_close(void) {
    if (s_open) { f_close(&s_fil); s_open = 0; }
}
