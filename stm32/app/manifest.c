/* manifest.c — đọc manifest.csv từng dòng bằng FatFS f_gets, parse CSV. */
#include "manifest.h"
#include "sdcard.h"
#include "board.h"

#include <stdlib.h>
#include <string.h>

static FIL  s_fil;
static int  s_open = 0;
static char s_line[160];

int manifest_open(const char *path) {
    if (s_open) { f_close(&s_fil); s_open = 0; }
    if (sdcard_open_read(&s_fil, path) != FR_OK) return -1;
    s_open = 1;
    /* bỏ dòng header */
    if (!f_gets(s_line, sizeof(s_line), &s_fil)) {
        uart_log("[manifest] empty file\r\n");
        return -2;
    }
    return 0;
}

static char *trim(char *s) {
    while (*s == ' ' || *s == '\t') s++;
    char *e = s + strlen(s);
    while (e > s && (e[-1] == '\r' || e[-1] == '\n' || e[-1] == ' ' || e[-1] == '\t'))
        *--e = '\0';
    return s;
}

int manifest_next(manifest_entry_t *out) {
    if (!s_open) return -1;
    /* bỏ qua dòng trống */
    for (;;) {
        if (!f_gets(s_line, sizeof(s_line), &s_fil)) return 0; /* EOF */
        char *l = trim(s_line);
        if (*l) break;
    }

    /* tách 7 trường bằng dấu phẩy */
    char *fields[7];
    int nf = 0;
    char *p = s_line;
    fields[nf++] = p;
    while (nf < 7 && (p = strchr(p, ',')) != NULL) {
        *p++ = '\0';
        fields[nf++] = p;
    }
    if (nf != 7) {
        uart_log("[manifest] bad line (%d fields)\r\n", nf);
        return -2;
    }

    memset(out, 0, sizeof(*out));
    strncpy(out->model_id,  trim(fields[0]), MODEL_ID_MAX - 1);
    strncpy(out->model_bin, trim(fields[1]), PATH_MAX_LEN - 1);
    out->input_w     = (uint32_t)strtoul(trim(fields[2]), NULL, 10);
    out->input_h     = (uint32_t)strtoul(trim(fields[3]), NULL, 10);
    out->input_c     = (uint32_t)strtoul(trim(fields[4]), NULL, 10);
    out->num_classes = (uint32_t)strtoul(trim(fields[5]), NULL, 10);
    out->crc32       = (uint32_t)strtoul(trim(fields[6]), NULL, 16); /* hex */
    return 1;
}

void manifest_close(void) {
    if (s_open) { f_close(&s_fil); s_open = 0; }
}
