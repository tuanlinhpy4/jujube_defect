/* main.h — khai báo chung, kéo HAL + handle peripheral cho cả Core và app/. */
#ifndef MAIN_H
#define MAIN_H

#include "stm32h7xx_hal.h"

/* Handle định nghĩa ở main.c, app/ tham chiếu qua extern riêng của từng file. */
extern UART_HandleTypeDef huart1;
extern QSPI_HandleTypeDef hqspi;
extern SD_HandleTypeDef   hsd1;

void Error_Handler(void);

#endif /* MAIN_H */
