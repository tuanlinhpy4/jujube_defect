/* board.c — DWT, MPU vùng QSPI, log UART. HAL chuẩn STM32H7. */
#include "board.h"
#include "model_ctx.h"

#include <stdarg.h>
#include <stdio.h>

#include "main.h"   /* CubeMX: kéo theo stm32h7xx_hal.h + extern handle */

/* Handle do CubeMX sinh trong main.c. Tên mặc định USART1 → huart1. */
extern UART_HandleTypeDef huart1;

/* Con trỏ tới thanh ghi DWT->CYCCNT cho dwt_cycles() inline (header không include core_cm7). */
volatile uint32_t *const _dwt_cyccnt = (volatile uint32_t *)0xE0001004u; /* DWT->CYCCNT */

/* RAM log mailbox — uart_log ghi song song vào đây để đọc qua ST-Link khi không có
 * COM/UART. magic="JLOG" báo buffer hợp lệ; pos = số byte đã ghi (dừng khi đầy). */
#define LOGBUF_SIZE 32768u
volatile uint32_t g_logmagic = 0;
volatile uint32_t g_logpos   = 0;
char g_logbuf[LOGBUF_SIZE];

void uart_log(const char *fmt, ...) {
    static char buf[256];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    if (n < 0) return;
    if (n > (int)sizeof(buf) - 1) n = (int)sizeof(buf) - 1; /* vsnprintf đã cắt + null */
    HAL_UART_Transmit(&huart1, (uint8_t *)buf, (uint16_t)n, 1000);
    /* mirror vào RAM mailbox */
    for (int i = 0; i < n && g_logpos < LOGBUF_SIZE; ++i)
        g_logbuf[g_logpos++] = buf[i];
    /* D-cache đang bật (CPU_CACHE_Enable trong main.c) → log kẹt trong cache, ST-Link
     * đọc SRAM vật lý ra 0. Clean prefix đã ghi để ST-Link thấy được log thật. */
    SCB_CleanDCache_by_Addr((uint32_t *)g_logbuf, (int32_t)g_logpos);
}

static void dwt_init(void) {
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

/* MPU: vùng QSPI 8MB (0x90000000) = Normal, CACHEABLE write-through, non-shareable.
 * Model read-only qua XIP → cacheable tăng tốc inference ~5-20× (weight reads vào D-cache).
 * Write đi qua QSPI indirect (bỏ qua cache); coherency đảm bảo bằng
 * board_qspi_cache_invalidate() sau mỗi lần program (xem qspi_model.c). */
static void mpu_config_qspi(void) {
    MPU_Region_InitTypeDef r = {0};
    HAL_MPU_Disable();

    r.Enable           = MPU_REGION_ENABLE;
    r.Number           = MPU_REGION_NUMBER0;
    r.BaseAddress      = QSPI_BASE_ADDR;       /* 0x90000000 */
    r.Size             = MPU_REGION_SIZE_8MB;
    r.SubRegionDisable = 0x00;
    r.TypeExtField     = MPU_TEX_LEVEL0;       /* TEX=000,C=1,B=0 → Normal write-through */
    r.AccessPermission = MPU_REGION_FULL_ACCESS;
    r.DisableExec      = MPU_INSTRUCTION_ACCESS_ENABLE; /* cho phép XIP */
    r.IsShareable      = MPU_ACCESS_NOT_SHAREABLE;      /* single-master → non-shareable cho cache */
    r.IsCacheable      = MPU_ACCESS_CACHEABLE;
    r.IsBufferable     = MPU_ACCESS_NOT_BUFFERABLE;     /* B=0: write-through */
    HAL_MPU_ConfigRegion(&r);

    HAL_MPU_Enable(MPU_PRIVILEGED_DEFAULT);
}

void board_qspi_cache_invalidate(uint32_t addr, uint32_t size) {
    /* dù MPU đã non-cacheable, vẫn invalidate cho chắc nếu user đổi cấu hình. */
    SCB_InvalidateDCache_by_Addr((uint32_t *)(addr & ~0x1Fu), (int32_t)(size + 32));
}

void board_init(void) {
    g_logmagic = 0x474F4C4Au;   /* "JLOG" — báo RAM mailbox hợp lệ */
    SCB_CleanDCache_by_Addr((uint32_t *)&g_logmagic, 4); /* để ST-Link đọc được magic */
    mpu_config_qspi();
    SCB_CleanInvalidateDCache();
    dwt_init();
    uart_log("\r\n========================================\r\n");
    uart_log("[board] STM32H750 jujube benchmark harness\r\n");
    extern uint32_t SystemCoreClock;
    uart_log("[board] SYSCLK=%lu Hz, DWT on, MPU QSPI cacheable (WT)\r\n",
             (unsigned long)SystemCoreClock);
}
