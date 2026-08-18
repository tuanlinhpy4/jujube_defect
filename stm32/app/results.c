/* results.c - extended benchmark, raw confusion matrix, and RAM CSV output. */
#include "results.h"
#include "sdcard.h"
#include "board.h"

#include <stdio.h>
#include <string.h>

static const char *BENCHMARK_HEADER =
    "model_id,clock_hz,num_samples,correct,accuracy,macro_precision,macro_recall,macro_f1,"
    "model_flash_bytes,peak_sram_bytes,tflm_arena_capacity_bytes,latency_avg_ms,"
    "latency_p95_ms\r\n";

static const char *CONFUSION_HEADER =
    "model_id,true_class,pred_class,count\r\n";

static const char *RAM_HEADER =
    "model_id,status,model_size_bytes,tflm_arena_used_bytes,tflm_arena_capacity_bytes,"
    "input_w,input_h,input_c,num_classes,note\r\n";

static int fmt_fixed(char *buf, int cap, float value, int decimals) {
    if (value < 0) value = 0;
    long scale = 1;
    for (int i = 0; i < decimals; ++i) scale *= 10;
    long scaled = (long)(value * (float)scale + 0.5f);
    long integer = scaled / scale;
    long fraction = scaled % scale;
    return snprintf(buf, cap, "%ld.%0*ld", integer, decimals, fraction);
}

static int open_append(FIL *file, const char *path, const char *header) {
    FRESULT fr = f_open(file, path, FA_WRITE | FA_OPEN_ALWAYS);
    if (fr != FR_OK) {
        uart_log("[results] open %s FAIL fr=%d\r\n", path, fr);
        return -1;
    }
    UINT written = 0;
    if (f_size(file) == 0) {
        fr = f_write(file, header, (UINT)strlen(header), &written);
        if (fr != FR_OK || written != (UINT)strlen(header)) {
            f_close(file);
            return -2;
        }
    }
    f_lseek(file, f_size(file));
    return 0;
}

int results_append_benchmark(const char *path, const char *model_id,
                             const benchmark_metrics_t *m,
                             uint32_t model_flash_bytes,
                             uint32_t peak_sram_bytes,
                             uint32_t arena_capacity_bytes) {
    if (!m) return -1;
    FIL file;
    if (open_append(&file, path, BENCHMARK_HEADER) != 0) return -1;

    char accuracy[16], precision[16], recall[16], f1[16], avg[16], p95[16];
    fmt_fixed(accuracy, sizeof(accuracy), m->accuracy, 4);
    fmt_fixed(precision, sizeof(precision), m->macro_precision, 4);
    fmt_fixed(recall, sizeof(recall), m->macro_recall, 4);
    fmt_fixed(f1, sizeof(f1), m->macro_f1, 4);
    fmt_fixed(avg, sizeof(avg), m->latency_avg_ms, 3);
    fmt_fixed(p95, sizeof(p95), m->latency_p95_ms, 3);

    char line[320];
    extern uint32_t SystemCoreClock;
    int n = snprintf(line, sizeof(line),
                     "%s,%lu,%lu,%lu,%s,%s,%s,%s,%lu,%lu,%lu,%s,%s\r\n",
                     model_id,
                     (unsigned long)SystemCoreClock,
                     (unsigned long)m->sample_count,
                     (unsigned long)m->correct_count,
                     accuracy, precision, recall, f1,
                     (unsigned long)model_flash_bytes,
                     (unsigned long)peak_sram_bytes,
                     (unsigned long)arena_capacity_bytes,
                     avg, p95);

    UINT written = 0;
    FRESULT fr = f_write(&file, line, (UINT)n, &written);
    f_close(&file);
    if (fr != FR_OK || written != (UINT)n) {
        uart_log("[results] benchmark write FAIL\r\n");
        return -2;
    }
    return 0;
}

int results_append_confusion(const char *path, const char *model_id,
                             const benchmark_metrics_t *m) {
    if (!m) return -1;
    FIL file;
    if (open_append(&file, path, CONFUSION_HEADER) != 0) return -1;

    for (uint32_t t = 0; t < m->num_classes; ++t) {
        for (uint32_t p = 0; p < m->num_classes; ++p) {
            char line[112];
            int n = snprintf(line, sizeof(line), "%s,%lu,%lu,%lu\r\n",
                             model_id, (unsigned long)t, (unsigned long)p,
                             (unsigned long)m->confusion[t][p]);
            UINT written = 0;
            FRESULT fr = f_write(&file, line, (UINT)n, &written);
            if (fr != FR_OK || written != (UINT)n) {
                f_close(&file);
                uart_log("[results] confusion write FAIL\r\n");
                return -2;
            }
        }
    }
    f_close(&file);
    return 0;
}

int results_append_ram(const char *path, const char *model_id,
                       const char *status, uint32_t model_size_bytes,
                       uint32_t tflm_arena_used_bytes,
                       uint32_t tflm_arena_capacity_bytes,
                       uint32_t input_w, uint32_t input_h,
                       uint32_t input_c, uint32_t num_classes,
                       const char *note) {
    FIL file;
    if (open_append(&file, path, RAM_HEADER) != 0) return -1;

    char line[224];
    int n = snprintf(line, sizeof(line), "%s,%s,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%s\r\n",
                     model_id, status,
                     (unsigned long)model_size_bytes,
                     (unsigned long)tflm_arena_used_bytes,
                     (unsigned long)tflm_arena_capacity_bytes,
                     (unsigned long)input_w, (unsigned long)input_h,
                     (unsigned long)input_c, (unsigned long)num_classes,
                     note ? note : "");

    UINT written = 0;
    FRESULT fr = f_write(&file, line, (UINT)n, &written);
    f_close(&file);
    if (fr != FR_OK || written != (UINT)n) {
        uart_log("[ram] write FAIL\r\n");
        return -2;
    }
    return 0;
}
