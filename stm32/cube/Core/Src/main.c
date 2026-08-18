/* WeAct MiniSTM32H750 benchmark bootstrap.
 * JUJUBE_SYSCLK_480 selects the STM32H750 maximum 480 MHz VOS0 profile.
 * SDMMC1 4-bit, W25Q64 QSPI, and USART1 are initialized before kernel_main(). */
#include "main.h"
#include "kernel.h"
#include "qspi_model.h"
#include "board.h"
#include "ff.h"

SD_HandleTypeDef hsd1;
QSPI_HandleTypeDef hqspi;
UART_HandleTypeDef huart1;

FATFS SDFatFS;
char SDPath[4] = "0:/";

static void SystemClock_Config(void);
static void PeriphCommonClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_USART1_UART_Init(void);
static void MX_QUADSPI_Init(void);
static void MX_SDMMC1_SD_Init(void);
static void CPU_CACHE_Enable(void);

int main(void) {
    CPU_CACHE_Enable();
    HAL_Init();
    SystemClock_Config();
    PeriphCommonClock_Config();

    MX_GPIO_Init();
    MX_USART1_UART_Init();
    MX_QUADSPI_Init();
    MX_SDMMC1_SD_Init();
    int app_load_rc = qspi_load_app_if_needed("0:/tflm_app.bin");
    if (app_load_rc != 0) {
        uart_log("[boot] qspi_load_app_if_needed FAIL rc=%d\r\n", app_load_rc);
        Error_Handler();
    }
    int memmap_rc = qspi_enable_memmap();
    if (memmap_rc != 0) {
        uart_log("[boot] qspi_enable_memmap FAIL rc=%d\r\n", memmap_rc);
        Error_Handler();
    }

    kernel_main();
    while (1) {}
}

static void SystemClock_Config(void) {
    RCC_ClkInitTypeDef clock = {0};
    RCC_OscInitTypeDef osc = {0};

    HAL_PWREx_ConfigSupply(PWR_LDO_SUPPLY);
    __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);
    while (!__HAL_PWR_GET_FLAG(PWR_FLAG_VOSRDY)) {}

#ifdef JUJUBE_SYSCLK_480
    /* ST reference sequence for H743/H750: VOS1 first, then VOS0. */
    __HAL_RCC_SYSCFG_CLK_ENABLE();
    __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE0);
    while (!__HAL_PWR_GET_FLAG(PWR_FLAG_VOSRDY)) {}
#endif

    osc.OscillatorType = RCC_OSCILLATORTYPE_HSE;
    osc.HSEState = RCC_HSE_ON;
    osc.HSIState = RCC_HSI_OFF;
    osc.CSIState = RCC_CSI_OFF;
    osc.PLL.PLLState = RCC_PLL_ON;
    osc.PLL.PLLSource = RCC_PLLSOURCE_HSE;
    osc.PLL.PLLM = 5; /* HSE 25 MHz / 5 = 5 MHz. */
#ifdef JUJUBE_SYSCLK_480
    osc.PLL.PLLN = 192; /* VCO 960 MHz, PLLP 480 MHz. */
    osc.PLL.PLLQ = 5;   /* SDMMC kernel 192 MHz. */
#else
    osc.PLL.PLLN = 160; /* VCO 800 MHz, PLLP 400 MHz. */
    osc.PLL.PLLQ = 4;   /* SDMMC kernel 200 MHz. */
#endif
    osc.PLL.PLLFRACN = 0;
    osc.PLL.PLLP = 2;
    osc.PLL.PLLR = 2;
    osc.PLL.PLLVCOSEL = RCC_PLL1VCOWIDE;
    osc.PLL.PLLRGE = RCC_PLL1VCIRANGE_2;
    if (HAL_RCC_OscConfig(&osc) != HAL_OK) Error_Handler();

    clock.ClockType = RCC_CLOCKTYPE_SYSCLK | RCC_CLOCKTYPE_HCLK |
                      RCC_CLOCKTYPE_D1PCLK1 | RCC_CLOCKTYPE_PCLK1 |
                      RCC_CLOCKTYPE_PCLK2 | RCC_CLOCKTYPE_D3PCLK1;
    clock.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
    clock.SYSCLKDivider = RCC_SYSCLK_DIV1;
    clock.AHBCLKDivider = RCC_HCLK_DIV2; /* 240 MHz at the 480 MHz profile. */
    clock.APB3CLKDivider = RCC_APB3_DIV2;
    clock.APB1CLKDivider = RCC_APB1_DIV2;
    clock.APB2CLKDivider = RCC_APB2_DIV2;
    clock.APB4CLKDivider = RCC_APB4_DIV2;
    if (HAL_RCC_ClockConfig(&clock, FLASH_LATENCY_4) != HAL_OK) Error_Handler();

#ifdef JUJUBE_SYSCLK_480
    /* Recommended by ST for high-speed GPIO/QSPI operation. */
    __HAL_RCC_CSI_ENABLE();
    __HAL_RCC_SYSCFG_CLK_ENABLE();
    HAL_EnableCompensationCell();
#endif
}

static void PeriphCommonClock_Config(void) {
    RCC_PeriphCLKInitTypeDef peripheral = {0};
    peripheral.PeriphClockSelection = RCC_PERIPHCLK_SDMMC | RCC_PERIPHCLK_QSPI;
    peripheral.SdmmcClockSelection = RCC_SDMMCCLKSOURCE_PLL;
    peripheral.QspiClockSelection = RCC_QSPICLKSOURCE_D1HCLK;
    if (HAL_RCCEx_PeriphCLKConfig(&peripheral) != HAL_OK) Error_Handler();
}

static void MX_USART1_UART_Init(void) {
    huart1.Instance = USART1;
    huart1.Init.BaudRate = 115200;
    huart1.Init.WordLength = UART_WORDLENGTH_8B;
    huart1.Init.StopBits = UART_STOPBITS_1;
    huart1.Init.Parity = UART_PARITY_NONE;
    huart1.Init.Mode = UART_MODE_TX_RX;
    huart1.Init.HwFlowCtl = UART_HWCONTROL_NONE;
    huart1.Init.OverSampling = UART_OVERSAMPLING_16;
    huart1.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
    huart1.Init.ClockPrescaler = UART_PRESCALER_DIV1;
    huart1.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
    if (HAL_UART_Init(&huart1) != HAL_OK) Error_Handler();
}

static void MX_QUADSPI_Init(void) {
    hqspi.Instance = QUADSPI;
#ifdef JUJUBE_SYSCLK_480
    hqspi.Init.ClockPrescaler = 4; /* 240/(4+1) = 48 MHz. */
#else
    hqspi.Init.ClockPrescaler = 3; /* 200/(3+1) = 50 MHz. */
#endif
    hqspi.Init.FifoThreshold = 4;
    hqspi.Init.SampleShifting = QSPI_SAMPLE_SHIFTING_HALFCYCLE;
    hqspi.Init.FlashSize = 22; /* 2^(22+1) = 8 MB W25Q64. */
    hqspi.Init.ChipSelectHighTime = QSPI_CS_HIGH_TIME_2_CYCLE;
    hqspi.Init.ClockMode = QSPI_CLOCK_MODE_0;
    hqspi.Init.FlashID = QSPI_FLASH_ID_1;
    hqspi.Init.DualFlash = QSPI_DUALFLASH_DISABLE;
    if (HAL_QSPI_Init(&hqspi) != HAL_OK) Error_Handler();
}

static void MX_SDMMC1_SD_Init(void) {
    hsd1.Instance = SDMMC1;
    hsd1.Init.ClockEdge = SDMMC_CLOCK_EDGE_RISING;
    hsd1.Init.ClockPowerSave = SDMMC_CLOCK_POWER_SAVE_DISABLE;
    hsd1.Init.BusWide = SDMMC_BUS_WIDE_1B;
    hsd1.Init.HardwareFlowControl = SDMMC_HARDWARE_FLOW_CONTROL_DISABLE;
    hsd1.Init.ClockDiv = 4; /* 24 MHz at SYSCLK 480 MHz (PLL1Q = 192 MHz). */
}

static void MX_GPIO_Init(void) {
    __HAL_RCC_GPIOA_CLK_ENABLE();
    __HAL_RCC_GPIOB_CLK_ENABLE();
    __HAL_RCC_GPIOC_CLK_ENABLE();
    __HAL_RCC_GPIOD_CLK_ENABLE();
    __HAL_RCC_GPIOE_CLK_ENABLE();
}

static void CPU_CACHE_Enable(void) {
    SCB_EnableICache();
    SCB_EnableDCache();
}

void Error_Handler(void) {
    __disable_irq();
    while (1) {}
}

#ifdef USE_FULL_ASSERT
void assert_failed(uint8_t *file, uint32_t line) {
    (void)file;
    (void)line;
}
#endif
