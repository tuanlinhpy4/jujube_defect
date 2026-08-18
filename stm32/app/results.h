/* results.h - append extended benchmark records to CSV files on SD. */
#ifndef RESULTS_H
#define RESULTS_H

#include <stdint.h>
#include "metrics.h"

int results_append_benchmark(const char *path, const char *model_id,
                             const benchmark_metrics_t *metrics,
                             uint32_t model_flash_bytes,
                             uint32_t peak_sram_bytes,
                             uint32_t arena_capacity_bytes);

int results_append_confusion(const char *path, const char *model_id,
                             const benchmark_metrics_t *metrics);

int results_append_ram(const char *path, const char *model_id,
                       const char *status, uint32_t model_size_bytes,
                       uint32_t tflm_arena_used_bytes,
                       uint32_t tflm_arena_capacity_bytes,
                       uint32_t input_w, uint32_t input_h,
                       uint32_t input_c, uint32_t num_classes,
                       const char *note);

#endif /* RESULTS_H */
