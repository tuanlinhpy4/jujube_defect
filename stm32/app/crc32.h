/* crc32.h — CRC-32/ISO-HDLC (zlib), khớp host zlib.crc32. Xem PACK_FORMAT.md §4. */
#ifndef CRC32_H
#define CRC32_H

#include <stddef.h>
#include <stdint.h>

/* CRC luỹ tiến: gọi lần đầu với crc=0. */
uint32_t crc32_update(uint32_t crc, const void *data, size_t len);

/* Tiện ích: CRC một buffer liền (vd vùng QSPI memory-mapped). */
static inline uint32_t crc32_buf(const void *data, size_t len) {
    return crc32_update(0u, data, len);
}

#endif /* CRC32_H */
