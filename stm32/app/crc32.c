/* crc32.c — CRC-32/ISO-HDLC reflected (poly 0xEDB88320, init/xorout 0xFFFFFFFF).
 * Khớp bit-for-bit với Python zlib.crc32 (host make_manifest.py). */
#include "crc32.h"

static uint32_t s_table[256];
static int s_table_ready = 0;

static void crc32_build_table(void) {
    for (uint32_t i = 0; i < 256; ++i) {
        uint32_t c = i;
        for (int k = 0; k < 8; ++k)
            c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
        s_table[i] = c;
    }
    s_table_ready = 1;
}

uint32_t crc32_update(uint32_t crc, const void *data, size_t len) {
    if (!s_table_ready) crc32_build_table();
    const uint8_t *p = (const uint8_t *)data;
    crc ^= 0xFFFFFFFFu;
    while (len--)
        crc = s_table[(crc ^ *p++) & 0xFFu] ^ (crc >> 8);
    return crc ^ 0xFFFFFFFFu;
}
