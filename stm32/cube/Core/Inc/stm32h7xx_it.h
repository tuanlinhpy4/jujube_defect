/* stm32h7xx_it.h — khai báo các exception handler Cortex-M7. */
#ifndef STM32H7xx_IT_H
#define STM32H7xx_IT_H

void NMI_Handler(void);
void HardFault_Handler(void);
void MemManage_Handler(void);
void BusFault_Handler(void);
void UsageFault_Handler(void);
void SVC_Handler(void);
void DebugMon_Handler(void);
void PendSV_Handler(void);
void SysTick_Handler(void);

#endif /* STM32H7xx_IT_H */
