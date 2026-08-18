/* metrics.h - classification metrics, confusion matrix, and latency stats. */
#ifndef METRICS_H
#define METRICS_H

#include <stdint.h>
#include "model_ctx.h"

typedef struct {
    uint32_t num_classes;
    uint32_t sample_count;
    uint32_t correct_count;
    float accuracy;
    float macro_precision;
    float macro_recall;
    float macro_f1;
    float latency_avg_ms;
    float latency_p95_ms;
    uint32_t confusion[JUJUBE_NUM_CLASSES][JUJUBE_NUM_CLASSES];
} benchmark_metrics_t;

void metrics_reset(uint32_t num_classes);
void metrics_update(uint32_t true_label, uint32_t pred_label, uint32_t latency_cycles);

/* Finalize all metrics. p95 follows nearest-rank: ceil(0.95*N), one-based. */
void metrics_finalize(benchmark_metrics_t *out);

#endif /* METRICS_H */
