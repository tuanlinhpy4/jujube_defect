/* inference.cc — BACKEND TFLite-Micro + CMSIS-NN (Phase 4).
 *
 * Thay STUB cũ. Đọc model .tflite trực tiếp tại ctx->slot_base (QSPI memory-mapped),
 * dựng MicroInterpreter với op-resolver = HỢP op của bộ model jujube, chạy 1 ảnh:
 *   input: pack float32 NCHW [C,H,W] (đã normalize) -> transpose sang NHWC float32
 *          (model IO float; Quantize/Dequantize nằm trong graph -> không tự lượng tử).
 *   output: logits float32 [num_classes].
 *
 * Arena đặt ở RAM_D1 (512KB, .tflm_arena) — system (.data/.bss/s_input) dời sang DTCM.
 * Giao tiếp giữ nguyên inference.h (kernel.c không đổi).
 *
 * Op-set: hợp bộ model cũ và supplemental = ADD AVERAGE_POOL_2D CONCATENATION
 *   CONV_2D DEPTHWISE_CONV_2D DEQUANTIZE FULLY_CONNECTED HARD_SWISH LOGISTIC
 *   MAX_POOL_2D MEAN MUL PAD QUANTIZE RESHAPE RESIZE_NEAREST_NEIGHBOR.
 *   Định nghĩa TFLM_OPS_MNASNET để chỉ đăng ký 8 op của mnasnet (build nhỏ, test Bước A).
 */
#include "inference.h"
#include "board.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_generated.h"

#ifndef TFLM_ARENA_BYTES
#define TFLM_ARENA_BYTES (504u * 1024u)   /* RAM_D1 512KB chiếm trọn (system dời sang DTCM) */
#endif

/* Arena: RAM_D1 512KB (xem linker .tflm_arena). System (.data/.bss/stack) ở DTCM. */
__attribute__((section(".tflm_arena"), aligned(16)))
static uint8_t s_arena[TFLM_ARENA_BYTES];

#ifdef TFLM_OPS_MNASNET
static constexpr int kNumOps = 8;
#else
static constexpr int kNumOps = 16;
#endif
using JujubeOpResolver = tflite::MicroMutableOpResolver<kNumOps>;

/* Bộ nhớ tĩnh cho resolver (1 lần) + interpreter (tái dựng mỗi model) — placement new,
 * không heap. */
alignas(8) static uint8_t s_resolver_storage[sizeof(JujubeOpResolver)];
alignas(8) static uint8_t s_interp_storage[sizeof(tflite::MicroInterpreter)];

static JujubeOpResolver       *s_resolver = nullptr;
static tflite::MicroInterpreter *s_interp = nullptr;
static uint32_t s_arena_used_bytes = 0;

static int build_resolver_once(void) {
    if (s_resolver) return 0;
    s_resolver = new (s_resolver_storage) JujubeOpResolver();
    TfLiteStatus st = kTfLiteOk;
    /* IO float wrapper + lõi conv */
    st = (st == kTfLiteOk) ? s_resolver->AddQuantize()        : st;
    st = (st == kTfLiteOk) ? s_resolver->AddDequantize()      : st;
    st = (st == kTfLiteOk) ? s_resolver->AddConv2D()          : st;
    st = (st == kTfLiteOk) ? s_resolver->AddDepthwiseConv2D() : st;
    st = (st == kTfLiteOk) ? s_resolver->AddFullyConnected()  : st;
    st = (st == kTfLiteOk) ? s_resolver->AddAdd()             : st;
    st = (st == kTfLiteOk) ? s_resolver->AddMean()            : st;
    st = (st == kTfLiteOk) ? s_resolver->AddPad()             : st;
#ifndef TFLM_OPS_MNASNET
    st = (st == kTfLiteOk) ? s_resolver->AddMul()             : st;
    st = (st == kTfLiteOk) ? s_resolver->AddConcatenation()   : st;
    st = (st == kTfLiteOk) ? s_resolver->AddLogistic()        : st;
    st = (st == kTfLiteOk) ? s_resolver->AddMaxPool2D()       : st;
    st = (st == kTfLiteOk) ? s_resolver->AddReshape()         : st;
    st = (st == kTfLiteOk) ? s_resolver->AddAveragePool2D()   : st;
    st = (st == kTfLiteOk) ? s_resolver->AddResizeNearestNeighbor() : st;
    st = (st == kTfLiteOk) ? s_resolver->AddHardSwish()       : st;
#endif
    if (st != kTfLiteOk) {
        uart_log("[infer] FATAL: dang ky op that bai\r\n");
        return -1;
    }
    return 0;
}

extern "C" int inference_init(const model_ctx_t *ctx) {
    s_arena_used_bytes = 0;
    uart_log("[infer] init start\r\n");
    if (build_resolver_once() != 0) return -1;
    uart_log("[infer] resolver OK\r\n");

    const tflite::Model *model = tflite::GetModel((const void *)(uintptr_t)ctx->slot_base);
    uart_log("[infer] GetModel OK\r\n");
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        uart_log("[infer] schema ver %lu != %d\r\n",
                 (unsigned long)model->version(), TFLITE_SCHEMA_VERSION);
        return -2;
    }
    uart_log("[infer] schema ver OK\r\n");

    /* KHÔNG gọi destructor interpreter cũ: model cũ tại slot QSPI (MODEL_SLOT_BASE) đã bị
     * model mới ghi đè TRƯỚC khi vào đây (kernel gọi qspi_load_model rồi mới inference_init).
     * ~MicroInterpreter()->FreeSubgraphs() sẽ deref model_ stale (giờ là data model mới) →
     * con trỏ hàm rác → HardFault IBUSERR. Không heap nên bỏ destructor không rò rỉ; arena
     * dùng chung được MicroAllocator của interpreter mới khởi tạo lại hoàn toàn. */
    uart_log("[infer] creating interpreter...\r\n");
    s_interp = new (s_interp_storage)
        tflite::MicroInterpreter(model, *s_resolver, s_arena, sizeof(s_arena));
    uart_log("[infer] interpreter OK\r\n");

    uart_log("[infer] AllocateTensors start...\r\n");
    if (s_interp->AllocateTensors() != kTfLiteOk) {
        uart_log("[infer] AllocateTensors FAIL (arena %u KB du khong?)\r\n",
                 (unsigned)(sizeof(s_arena) / 1024));
        return -3;
    }
    uart_log("[infer] AllocateTensors OK\r\n");
    s_arena_used_bytes = (uint32_t)s_interp->arena_used_bytes();

    TfLiteTensor *in = s_interp->input(0);
    TfLiteTensor *out = s_interp->output(0);
    uart_log("[infer] %s arena_used=%u/%u KB  in[type=%d dims=%d:%dx%dx%d] out[type=%d n=%d]\r\n",
             ctx->model_id,
             (unsigned)(s_arena_used_bytes / 1024),
             (unsigned)(sizeof(s_arena) / 1024),
             in->type, in->dims->size,
             in->dims->size >= 4 ? in->dims->data[1] : 0,
             in->dims->size >= 4 ? in->dims->data[2] : 0,
             in->dims->size >= 4 ? in->dims->data[3] : 0,
             out->type, out->dims->data[out->dims->size - 1]);

    if (in->type != kTfLiteFloat32 || out->type != kTfLiteFloat32) {
        uart_log("[infer] WARN: IO khong phai float32 (in=%d out=%d)\r\n",
                 in->type, out->type);
    }
    return 0;
}

extern "C" uint32_t inference_arena_used_bytes(void) {
    return s_arena_used_bytes;
}

extern "C" uint32_t inference_arena_capacity_bytes(void) {
    return (uint32_t)sizeof(s_arena);
}

extern "C" int inference_run(const model_ctx_t *ctx, const float *input, float *output) {
    if (!s_interp) return -1;
    TfLiteTensor *in = s_interp->input(0);

    /* pack: NCHW [c*H*W + h*W + w]  ->  tensor NHWC [(h*W+w)*C + c] */
    const int H = (int)ctx->input_h, W = (int)ctx->input_w, C = (int)ctx->input_c;
    float *dst = in->data.f;
    for (int c = 0; c < C; ++c) {
        const float *plane = input + (size_t)c * H * W;
        for (int hw = 0; hw < H * W; ++hw) {
            dst[hw * C + c] = plane[hw];
        }
    }

    if (s_interp->Invoke() != kTfLiteOk) {
        uart_log("[infer] Invoke FAIL %s\r\n", ctx->model_id);
        return -2;
    }

    const float *src = s_interp->output(0)->data.f;
    for (uint32_t i = 0; i < ctx->num_classes; ++i) output[i] = src[i];
    return 0;
}
