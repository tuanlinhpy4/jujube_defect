/* inference.h — interface infer 2 backend: STUB (Phase 3) / TFLM+CMSIS-NN (Phase 4).
 * Không hardcode để swap dễ. Xem PLAN.md §8. */
#ifndef INFERENCE_H
#define INFERENCE_H

#include "model_ctx.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Chuẩn bị backend cho model trong ctx (STUB: noop; TFLM: dựng interpreter từ
 * ctx->slot_base). Trả 0 OK. */
int inference_init(const model_ctx_t *ctx);

/* TFLM arena bytes used after AllocateTensors().
 * Returns 0 before a successful inference_init(). */
uint32_t inference_arena_used_bytes(void);

/* Tensor arena capacity compiled into the firmware. */
uint32_t inference_arena_capacity_bytes(void);

/* Chạy 1 ảnh: input[C*H*W] float NCHW → output[num_classes] logits. Trả 0 OK. */
int inference_run(const model_ctx_t *ctx, const float *input, float *output);

#ifdef __cplusplus
}
#endif

#endif /* INFERENCE_H */
