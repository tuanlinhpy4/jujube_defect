/* FatFs glue for the polling STM32H7 SDMMC HAL driver. */
#include "diskio.h"
#include "ff.h"
#include "main.h"
#include "board.h"

#include <string.h>

extern SD_HandleTypeDef hsd1;

#define SD_TIMEOUT_MS 2000u
static volatile DSTATUS s_stat = STA_NOINIT;
static uint8_t s_scratch[512] __attribute__((aligned(32)));

static int sd_wait_ready(uint32_t timeout_ms) {
    uint32_t start = HAL_GetTick();
    while (HAL_SD_GetCardState(&hsd1) != HAL_SD_CARD_TRANSFER) {
        if (HAL_GetTick() - start > timeout_ms) return -1;
    }
    return 0;
}

static int sd_read_sector(uint32_t sector) {
    if (HAL_SD_ReadBlocks(&hsd1, s_scratch, sector, 1, SD_TIMEOUT_MS) == HAL_OK &&
        sd_wait_ready(SD_TIMEOUT_MS) == 0) {
        return 0;
    }
    uart_log("[sdio] read sector=%lu FAIL err=0x%08lX\r\n",
             (unsigned long)sector, (unsigned long)HAL_SD_GetError(&hsd1));
    return -1;
}

#if _FS_READONLY == 0
static int sd_write_sector(uint32_t sector) {
    if (HAL_SD_WriteBlocks(&hsd1, s_scratch, sector, 1, SD_TIMEOUT_MS) == HAL_OK &&
        sd_wait_ready(SD_TIMEOUT_MS) == 0) {
        return 0;
    }
    uart_log("[sdio] write sector=%lu FAIL err=0x%08lX\r\n",
             (unsigned long)sector, (unsigned long)HAL_SD_GetError(&hsd1));
    return -1;
}
#endif

DSTATUS disk_status(BYTE pdrv) {
    (void)pdrv;
    return s_stat;
}

DSTATUS disk_initialize(BYTE pdrv) {
    (void)pdrv;
    s_stat = STA_NOINIT;
    HAL_StatusTypeDef status = HAL_SD_Init(&hsd1);
    if (status != HAL_OK) {
        uart_log("[sdio] HAL_SD_Init FAIL st=%d err=0x%08lX\r\n",
                 (int)status, (unsigned long)HAL_SD_GetError(&hsd1));
        return s_stat;
    }
    status = HAL_SD_ConfigWideBusOperation(&hsd1, SDMMC_BUS_WIDE_4B);
    if (status != HAL_OK) {
        uart_log("[sdio] 4-bit FAIL st=%d err=0x%08lX -> keep 1-bit\r\n",
                 (int)status, (unsigned long)HAL_SD_GetError(&hsd1));
    } else {
        uart_log("[sdio] 4-bit, 24MHz\r\n");
    }
    if (sd_wait_ready(SD_TIMEOUT_MS) == 0) {
        s_stat &= ~STA_NOINIT;
    } else {
        uart_log("[sdio] card not ready state=%lu err=0x%08lX\r\n",
                 (unsigned long)HAL_SD_GetCardState(&hsd1),
                 (unsigned long)HAL_SD_GetError(&hsd1));
    }
    return s_stat;
}

DRESULT disk_read(BYTE pdrv, BYTE *buffer, DWORD sector, UINT count) {
    (void)pdrv;
    if (s_stat & STA_NOINIT) return RES_NOTRDY;
    for (UINT i = 0; i < count; ++i) {
        if (sd_read_sector((uint32_t)(sector + i)) != 0) return RES_ERROR;
        memcpy(buffer + (size_t)i * 512u, s_scratch, 512u);
    }
    return RES_OK;
}

#if _FS_READONLY == 0
DRESULT disk_write(BYTE pdrv, const BYTE *buffer, DWORD sector, UINT count) {
    (void)pdrv;
    if (s_stat & STA_NOINIT) return RES_NOTRDY;
    for (UINT i = 0; i < count; ++i) {
        memcpy(s_scratch, buffer + (size_t)i * 512u, 512u);
        if (sd_write_sector((uint32_t)(sector + i)) != 0) return RES_ERROR;
    }
    return RES_OK;
}
#endif

DRESULT disk_ioctl(BYTE pdrv, BYTE command, void *buffer) {
    (void)pdrv;
    if (s_stat & STA_NOINIT) return RES_NOTRDY;
    HAL_SD_CardInfoTypeDef info;
    switch (command) {
        case CTRL_SYNC:
            return RES_OK;
        case GET_SECTOR_COUNT:
            HAL_SD_GetCardInfo(&hsd1, &info);
            *(DWORD *)buffer = info.LogBlockNbr;
            return RES_OK;
        case GET_SECTOR_SIZE:
            *(WORD *)buffer = 512;
            return RES_OK;
        case GET_BLOCK_SIZE:
            *(DWORD *)buffer = 1;
            return RES_OK;
        default:
            return RES_PARERR;
    }
}

DWORD get_fattime(void) {
    return ((DWORD)(2024 - 1980) << 25) | ((DWORD)1 << 21) | ((DWORD)1 << 16);
}
