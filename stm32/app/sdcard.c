/* sdcard.c — FatFS mount + helper. Dựa trên object CubeMX sinh trong fatfs.c. */
#include "sdcard.h"
#include "board.h"

/* CubeMX (fatfs.c) sinh sẵn: */
extern FATFS SDFatFS;   /* file system object */
extern char  SDPath[4]; /* logical drive path, vd "0:/" */

static int s_mounted = 0;

int sdcard_mount(void) {
    if (s_mounted) {
        uart_log("[sd] already mounted '%s'\r\n", SDPath);
        return 0;
    }
    FRESULT fr = f_mount(&SDFatFS, SDPath, 1 /* mount ngay */);
    if (fr != FR_OK) {
        uart_log("[sd] f_mount FAIL fr=%d (path='%s')\r\n", fr, SDPath);
        return (int)fr;
    }

    FATFS *fs;
    DWORD free_clusters = 0;
    if (f_getfree(SDPath, &free_clusters, &fs) == FR_OK) {
        uint64_t total_kb = (uint64_t)(fs->n_fatent - 2) * fs->csize / 2;
        uint64_t free_kb  = (uint64_t)free_clusters * fs->csize / 2;
        uart_log("[sd] mounted '%s'  total=%lu MB  free=%lu MB\r\n",
                 SDPath, (unsigned long)(total_kb / 1024),
                 (unsigned long)(free_kb / 1024));
    } else {
        uart_log("[sd] mounted '%s' (f_getfree failed)\r\n", SDPath);
    }
    s_mounted = 1;
    return 0;
}

FRESULT sdcard_open_read(FIL *fp, const char *path) {
    FRESULT fr = f_open(fp, path, FA_READ);
    if (fr != FR_OK)
        uart_log("[sd] open '%s' FAIL fr=%d\r\n", path, fr);
    return fr;
}

FRESULT sdcard_read_at(FIL *fp, uint32_t ofs, void *buf, uint32_t len, UINT *br) {
    FRESULT fr = f_lseek(fp, ofs);
    if (fr != FR_OK) return fr;
    return f_read(fp, buf, len, br);
}
