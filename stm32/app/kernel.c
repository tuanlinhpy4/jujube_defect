/* kernel.c - SD -> QSPI -> TFLM benchmark orchestrator. */
#include "kernel.h"
#include "board.h"
#include "sdcard.h"
#include "qspi_model.h"
#include "manifest.h"
#include "test_pack.h"
#include "metrics.h"
#include "results.h"
#include "inference.h"
#include "model_ctx.h"

#include "main.h"
#include "ff.h"

#define PATH_MANIFEST  "manifest.csv"
#define PATH_TESTPACK  "test/test_data.pack"
#define PATH_RESULTS   "results/benchmark_metrics.csv"
#define PATH_CONFUSION "results/confusion_matrix.csv"
#define PATH_RAM_USAGE "results/ram_usage.csv"

static float s_input[JUJUBE_INPUT_FLOATS];
static float s_output[JUJUBE_NUM_CLASSES];

#ifndef JUJUBE_RAM_ONLY
static uint32_t argmax(const float *values, uint32_t count) {
    uint32_t best = 0;
    for (uint32_t i = 1; i < count; ++i) {
        if (values[i] > values[best]) best = i;
    }
    return best;
}
#endif

static void halt(const char *reason) {
    uart_log("[kernel] HALT: %s\r\n", reason);
    for (;;) HAL_Delay(1000);
}

void kernel_main(void) {
    board_init();

#ifdef JUJUBE_SYSCLK_480
    extern uint32_t SystemCoreClock;
    if (SystemCoreClock != 480000000u) halt("SYSCLK is not 480 MHz");
#endif

    if (sdcard_mount() != 0) halt("SD mount");
    if (qspi_init() != 0) halt("QSPI id");
    f_mkdir("results");

    if (manifest_open(PATH_MANIFEST) != 0) halt("manifest open");

#ifdef JUJUBE_RAM_ONLY
    uart_log("[kernel] === RAM allocation pass ===\r\n");
#else
    uart_log("[kernel] === extended benchmark pass ===\r\n");
#endif

    manifest_entry_t entry;
    model_ctx_t ctx;
    int completed = 0;
    int manifest_rc;

    while ((manifest_rc = manifest_next(&entry)) == 1) {
        uart_log("\r\n[kernel] --- model: %s (%s) ---\r\n",
                 entry.model_id, entry.model_bin);

        for (int i = 0; i < MODEL_ID_MAX; ++i) ctx.model_id[i] = entry.model_id[i];
        ctx.input_w = entry.input_w;
        ctx.input_h = entry.input_h;
        ctx.input_c = entry.input_c;
        ctx.num_classes = entry.num_classes;
        ctx.crc32_expected = entry.crc32;
        ctx.slot_base = MODEL_SLOT_BASE;
        ctx.input_buf = s_input;
        ctx.output_buf = s_output;

        if (qspi_load_model(entry.model_bin, MODEL_SLOT_OFFSET, entry.crc32,
                            &ctx.model_size) != 0) {
            uart_log("[kernel] SKIP %s (load/CRC fail)\r\n", entry.model_id);
#ifdef JUJUBE_RAM_ONLY
            results_append_ram(PATH_RAM_USAGE, entry.model_id, "LOAD_FAIL",
                               0, 0, inference_arena_capacity_bytes(),
                               entry.input_w, entry.input_h, entry.input_c,
                               entry.num_classes, "load_or_crc_fail");
#endif
            continue;
        }

#ifdef JUJUBE_RAM_ONLY
        int init_rc = inference_init(&ctx);
        uint32_t arena_used = inference_arena_used_bytes();
        uint32_t arena_capacity = inference_arena_capacity_bytes();
        const char *status = init_rc == 0 ? "OK" : "INIT_FAIL";
        const char *note = init_rc == 0 ? "" : "AllocateTensors_or_schema_fail";
        results_append_ram(PATH_RAM_USAGE, entry.model_id, status,
                           ctx.model_size, arena_used, arena_capacity,
                           entry.input_w, entry.input_h, entry.input_c,
                           entry.num_classes, note);
        uart_log("[kernel] RAM %s status=%s model=%luB arena=%lu/%luB\r\n",
                 entry.model_id, status,
                 (unsigned long)ctx.model_size,
                 (unsigned long)arena_used,
                 (unsigned long)arena_capacity);
        if (init_rc == 0) completed++;
        continue;
#else
        if (test_pack_open(PATH_TESTPACK) != 0) halt("test_pack open");
        if (inference_init(&ctx) != 0) {
            test_pack_close();
            continue;
        }

        metrics_reset(entry.num_classes);
        uint32_t sample_count = test_pack_count();
        for (uint32_t i = 0; i < sample_count; ++i) {
            uint32_t label;
            if (test_pack_read(i, s_input, &label) != 0) halt("pack read");

            uint32_t start_cycles = dwt_cycles();
            int infer_rc = inference_run(&ctx, s_input, s_output);
            uint32_t elapsed_cycles = dwt_cycles() - start_cycles;
            if (infer_rc != 0) halt("TFLM Invoke");

            uint32_t prediction = argmax(s_output, entry.num_classes);
            metrics_update(label, prediction, elapsed_cycles);
            if ((i + 1u) % 100u == 0u) {
                uart_log("[kernel]   %lu/%lu\r\n",
                         (unsigned long)(i + 1u),
                         (unsigned long)sample_count);
            }
        }
        test_pack_close();

        benchmark_metrics_t summary;
        metrics_finalize(&summary);
        uint32_t arena_used = inference_arena_used_bytes();
        uint32_t arena_capacity = inference_arena_capacity_bytes();
        if (results_append_benchmark(PATH_RESULTS, entry.model_id, &summary,
                                     ctx.model_size, arena_used, arena_capacity) != 0) {
            halt("benchmark result write");
        }
        if (results_append_confusion(PATH_CONFUSION, entry.model_id, &summary) != 0) {
            halt("confusion result write");
        }

        uart_log("[kernel] DONE %s acc=%ld prec=%ld rec=%ld f1=%ld /10000 "
                 "lat_avg=%ldus p95=%ldus flash=%luB peak_sram=%luB\r\n",
                 entry.model_id,
                 (long)(summary.accuracy * 10000.0f + 0.5f),
                 (long)(summary.macro_precision * 10000.0f + 0.5f),
                 (long)(summary.macro_recall * 10000.0f + 0.5f),
                 (long)(summary.macro_f1 * 10000.0f + 0.5f),
                 (long)(summary.latency_avg_ms * 1000.0f + 0.5f),
                 (long)(summary.latency_p95_ms * 1000.0f + 0.5f),
                 (unsigned long)ctx.model_size,
                 (unsigned long)arena_used);
        completed++;
#endif
    }
    manifest_close();

    if (manifest_rc < 0) uart_log("[kernel] manifest parse error rc=%d\r\n", manifest_rc);
#ifdef JUJUBE_RAM_ONLY
    uart_log("\r\n[kernel] === done: %d models -> %s ===\r\n",
             completed, PATH_RAM_USAGE);
#else
    uart_log("\r\n[kernel] === done: %d models -> %s + %s ===\r\n",
             completed, PATH_RESULTS, PATH_CONFUSION);
#endif
    halt("done");
}
