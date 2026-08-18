/* stm32h7xx_hal_msp.c — MSP init theo pin board WeAct MiniSTM32H750.
 * USART1: PA9/PA10 (AF7). QUADSPI: PB2 CLK, PB6 NCS, PD11/12 IO0/1, PE2 IO2, PD13 IO3.
 * SDMMC1 4-bit: PC8-11 D0-3, PC12 CK, PD2 CMD (AF12). */
#include "main.h"

void HAL_MspInit(void) {
    __HAL_RCC_SYSCFG_CLK_ENABLE();
}

void HAL_UART_MspInit(UART_HandleTypeDef *huart) {
    if (huart->Instance != USART1) return;
    GPIO_InitTypeDef g = {0};
    __HAL_RCC_USART1_CLK_ENABLE();
    __HAL_RCC_GPIOA_CLK_ENABLE();
    /* PA9 = TX, PA10 = RX */
    g.Pin       = GPIO_PIN_9 | GPIO_PIN_10;
    g.Mode      = GPIO_MODE_AF_PP;
    g.Pull      = GPIO_PULLUP;
    g.Speed     = GPIO_SPEED_FREQ_VERY_HIGH;
    g.Alternate = GPIO_AF7_USART1;
    HAL_GPIO_Init(GPIOA, &g);
}

void HAL_QSPI_MspInit(QSPI_HandleTypeDef *hqspi_) {
    (void)hqspi_;
    GPIO_InitTypeDef g = {0};
    __HAL_RCC_QSPI_CLK_ENABLE();
    __HAL_RCC_GPIOB_CLK_ENABLE();
    __HAL_RCC_GPIOD_CLK_ENABLE();
    __HAL_RCC_GPIOE_CLK_ENABLE();

    /* PB2 = CLK (AF9) */
    g.Pin = GPIO_PIN_2; g.Mode = GPIO_MODE_AF_PP; g.Pull = GPIO_NOPULL;
    g.Speed = GPIO_SPEED_FREQ_VERY_HIGH; g.Alternate = GPIO_AF9_QUADSPI;
    HAL_GPIO_Init(GPIOB, &g);
    /* PB6 = BK1_NCS (AF10) */
    g.Pin = GPIO_PIN_6; g.Pull = GPIO_PULLUP; g.Alternate = GPIO_AF10_QUADSPI;
    HAL_GPIO_Init(GPIOB, &g);
    /* PD11 IO0, PD12 IO1, PD13 IO3 (AF9) */
    g.Pin = GPIO_PIN_11 | GPIO_PIN_12 | GPIO_PIN_13;
    g.Pull = GPIO_NOPULL; g.Alternate = GPIO_AF9_QUADSPI;
    HAL_GPIO_Init(GPIOD, &g);
    /* PE2 IO2 (AF9) */
    g.Pin = GPIO_PIN_2; g.Alternate = GPIO_AF9_QUADSPI;
    HAL_GPIO_Init(GPIOE, &g);
}

void HAL_SD_MspInit(SD_HandleTypeDef *hsd) {
    if (hsd->Instance != SDMMC1) return;
    GPIO_InitTypeDef g = {0};
    __HAL_RCC_SDMMC1_CLK_ENABLE();
    __HAL_RCC_GPIOC_CLK_ENABLE();
    __HAL_RCC_GPIOD_CLK_ENABLE();

    /* PC8-PC11 = D0-D3, PC12 = CK (AF12) */
    g.Pin       = GPIO_PIN_8 | GPIO_PIN_9 | GPIO_PIN_10 | GPIO_PIN_11 | GPIO_PIN_12;
    g.Mode      = GPIO_MODE_AF_PP;
    g.Pull      = GPIO_PULLUP;
    g.Speed     = GPIO_SPEED_FREQ_VERY_HIGH;
    g.Alternate = GPIO_AF12_SDMMC1;
    HAL_GPIO_Init(GPIOC, &g);
    /* PD2 = CMD (AF12) */
    g.Pin = GPIO_PIN_2;
    HAL_GPIO_Init(GPIOD, &g);
}
