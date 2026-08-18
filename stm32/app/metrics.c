/* metrics.c - confusion matrix, classification metrics, avg and p95 latency. */
#include "metrics.h"
#include "board.h"

#include <stdlib.h>
#include <string.h>

#define MAX_SAMPLES 2048u /* >= 1740 samples in the public dataset. */

static uint32_t s_nc;
static uint32_t s_cm[JUJUBE_NUM_CLASSES][JUJUBE_NUM_CLASSES];
static uint32_t s_lat[MAX_SAMPLES];
static uint32_t s_count;
static uint32_t s_correct;

void metrics_reset(uint32_t num_classes) {
    s_nc = (num_classes <= JUJUBE_NUM_CLASSES) ? num_classes : JUJUBE_NUM_CLASSES;
    s_count = 0;
    s_correct = 0;
    memset(s_cm, 0, sizeof(s_cm));
}

void metrics_update(uint32_t t, uint32_t p, uint32_t latency_cycles) {
    if (t < s_nc && p < s_nc) {
        s_cm[t][p]++;
        if (t == p) s_correct++;
    }
    if (s_count < MAX_SAMPLES) s_lat[s_count] = latency_cycles;
    s_count++;
}

static int cmp_u32(const void *a, const void *b) {
    uint32_t x = *(const uint32_t *)a;
    uint32_t y = *(const uint32_t *)b;
    return (x > y) - (x < y);
}

void metrics_finalize(benchmark_metrics_t *out) {
    if (!out) return;
    memset(out, 0, sizeof(*out));
    out->num_classes = s_nc;
    out->sample_count = s_count;
    out->correct_count = s_correct;
    out->accuracy = s_count ? (float)s_correct / (float)s_count : 0.0f;

    float precision_sum = 0.0f;
    float recall_sum = 0.0f;
    float f1_sum = 0.0f;
    for (uint32_t c = 0; c < s_nc; ++c) {
        uint32_t tp = s_cm[c][c];
        uint32_t fp = 0;
        uint32_t fn = 0;
        for (uint32_t k = 0; k < s_nc; ++k) {
            if (k != c) {
                fp += s_cm[k][c];
                fn += s_cm[c][k];
            }
        }
        float precision = (tp + fp) ? (float)tp / (float)(tp + fp) : 0.0f;
        float recall = (tp + fn) ? (float)tp / (float)(tp + fn) : 0.0f;
        float f1 = (precision + recall) > 0.0f
                     ? 2.0f * precision * recall / (precision + recall)
                     : 0.0f;
        precision_sum += precision;
        recall_sum += recall;
        f1_sum += f1;
    }
    if (s_nc) {
        out->macro_precision = precision_sum / (float)s_nc;
        out->macro_recall = recall_sum / (float)s_nc;
        out->macro_f1 = f1_sum / (float)s_nc;
    }

    uint32_t measured = (s_count < MAX_SAMPLES) ? s_count : MAX_SAMPLES;
    if (measured) {
        uint64_t total_cycles = 0;
        for (uint32_t i = 0; i < measured; ++i) total_cycles += s_lat[i];
        out->latency_avg_ms = dwt_cycles_to_ms((uint32_t)(total_cycles / measured));

        qsort(s_lat, measured, sizeof(s_lat[0]), cmp_u32);
        uint32_t rank = (95u * measured + 99u) / 100u; /* ceil(0.95*N) */
        uint32_t index = rank ? rank - 1u : 0u;
        if (index >= measured) index = measured - 1u;
        out->latency_p95_ms = dwt_cycles_to_ms(s_lat[index]);
    }

    memcpy(out->confusion, s_cm, sizeof(s_cm));
}
