/* board.h - DWT timing, QSPI MPU configuration, and UART logging. */
#ifndef BOARD_H
#define BOARD_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef JUJUBE_SYSCLK_480
#define SYSCLK_HZ 480000000u
#else
#define SYSCLK_HZ 400000000u
#endif

void board_init(void);
void uart_log(const char *fmt, ...);

static inline uint32_t dwt_cycles(void) {
    extern volatile uint32_t *const _dwt_cyccnt;
    return *_dwt_cyccnt;
}

/* Uses the actual HAL-maintained clock, not a hard-coded frequency. */
static inline float dwt_cycles_to_ms(uint32_t cycles) {
    extern uint32_t SystemCoreClock;
    return (float)cycles * 1000.0f / (float)SystemCoreClock;
}

void board_qspi_cache_invalidate(uint32_t addr, uint32_t size);

#ifdef __cplusplus
}
#endif

#endif /* BOARD_H */
