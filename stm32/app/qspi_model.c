/* qspi_model.c — driver W25Q64 (Winbond, 8MB) tối thiểu trên QUADSPI HAL.
 *
 * Bring-up an toàn: program PAGE_PROGRAM 1-1-1 (256B/page), memory-mapped read
 * FAST_READ 0x0B 1-1-1 (không cần set QE). Tối ưu quad (0x6B/0xEB) để Phase sau.
 * PHẢI abort memmap trước khi erase/program (không thể ghi khi đang XIP/mem-map đọc).
 */
#include "qspi_model.h"
#include "board.h"
#include "model_ctx.h"
#include "crc32.h"
#include "sdcard.h"

#include "main.h"
#include <string.h>

extern QSPI_HandleTypeDef hqspi;   /* CubeMX */

/* --- W25Q64 --- */
#define CMD_WRITE_ENABLE      0x06
#define CMD_READ_SR1          0x05
#define CMD_READ_JEDEC_ID     0x9F
#define CMD_BLOCK_ERASE_64K   0xD8
#define CMD_PAGE_PROGRAM      0x02
#define CMD_FAST_READ         0x0B
#define SR1_WIP               0x01

#define W25_PAGE_SIZE         256u
#define W25_BLOCK_SIZE        0x10000u   /* 64 KB */
#define QSPI_TMO              2000u      /* ms */

/* buffer streaming SD→QSPI (4KB, bội số page). Static để khỏi ăn stack. */
static uint8_t s_io[4096];

static int qspi_cmd_simple(uint8_t instr) {
    QSPI_CommandTypeDef c = {0};
    c.InstructionMode = QSPI_INSTRUCTION_1_LINE;
    c.Instruction     = instr;
    c.AddressMode     = QSPI_ADDRESS_NONE;
    c.DataMode        = QSPI_DATA_NONE;
    c.DummyCycles     = 0;
    return HAL_QSPI_Command(&hqspi, &c, QSPI_TMO) == HAL_OK ? 0 : -1;
}

static int qspi_write_enable(void) { return qspi_cmd_simple(CMD_WRITE_ENABLE); }

static int qspi_read_data(uint32_t flash_addr, uint8_t *data, uint32_t len) {
    if (len == 0) return 0;
    QSPI_CommandTypeDef c = {0};
    c.InstructionMode = QSPI_INSTRUCTION_1_LINE;
    c.Instruction     = CMD_FAST_READ;
    c.AddressMode     = QSPI_ADDRESS_1_LINE;
    c.AddressSize     = QSPI_ADDRESS_24_BITS;
    c.Address         = flash_addr;
    c.DataMode        = QSPI_DATA_1_LINE;
    c.DummyCycles     = 8;
    c.NbData          = len;
    if (HAL_QSPI_Command(&hqspi, &c, QSPI_TMO) != HAL_OK) return -1;
    return HAL_QSPI_Receive(&hqspi, data, QSPI_TMO) == HAL_OK ? 0 : -2;
}

static int qspi_crc_indirect(uint32_t flash_ofs, uint32_t size, uint32_t *out_crc) {
    uint32_t crc = 0;
    uint32_t done = 0;
    while (done < size) {
        uint32_t n = size - done;
        if (n > sizeof(s_io)) n = sizeof(s_io);
        if (qspi_read_data(flash_ofs + done, s_io, n) != 0) return -1;
        crc = crc32_update(crc, s_io, n);
        done += n;
    }
    *out_crc = crc;
    return 0;
}

static int file_crc(FIL *f, uint32_t *out_crc) {
    FRESULT fr = f_lseek(f, 0);
    if (fr != FR_OK) {
        uart_log("[qspi] file_crc initial seek FAIL fr=%d\r\n", fr);
        return -1;
    }
    uint32_t crc = 0;
    UINT br;
    while (1) {
        fr = f_read(f, s_io, sizeof(s_io), &br);
        if (fr != FR_OK) {
            uart_log("[qspi] file_crc read FAIL fr=%d ferr=%d\r\n", fr, f_error(f));
            return -2;
        }
        if (br == 0) break;
        crc = crc32_update(crc, s_io, br);
    }
    fr = f_lseek(f, 0);
    if (fr != FR_OK) {
        uart_log("[qspi] file_crc rewind FAIL fr=%d\r\n", fr);
        return -3;
    }
    *out_crc = crc;
    return 0;
}

static int qspi_wait_wip(uint32_t tmo_ms) {
    QSPI_CommandTypeDef c = {0};
    c.InstructionMode = QSPI_INSTRUCTION_1_LINE;
    c.Instruction     = CMD_READ_SR1;
    c.DataMode        = QSPI_DATA_1_LINE;
    c.NbData          = 1;
    uint32_t t = 0;
    uint8_t sr;
    do {
        if (HAL_QSPI_Command(&hqspi, &c, QSPI_TMO) != HAL_OK) return -1;
        if (HAL_QSPI_Receive(&hqspi, &sr, QSPI_TMO) != HAL_OK) return -1;
        if (!(sr & SR1_WIP)) return 0;
        HAL_Delay(1);
    } while (++t < tmo_ms);
    return -2; /* timeout */
}

int qspi_init(void) {
    HAL_QSPI_Abort(&hqspi);   /* thoát memmap nếu đang bật */
    QSPI_CommandTypeDef c = {0};
    c.InstructionMode = QSPI_INSTRUCTION_1_LINE;
    c.Instruction     = CMD_READ_JEDEC_ID;
    c.DataMode        = QSPI_DATA_1_LINE;
    c.NbData          = 3;
    uint8_t id[3] = {0};
    if (HAL_QSPI_Command(&hqspi, &c, QSPI_TMO) != HAL_OK) return -1;
    if (HAL_QSPI_Receive(&hqspi, id, QSPI_TMO) != HAL_OK) return -1;
    uart_log("[qspi] JEDEC ID = %02X %02X %02X\r\n", id[0], id[1], id[2]);
    /* Winbond=0xEF, W25Q64 type=0x40, capacity=0x17 */
    if (id[0] != 0xEF) {
        uart_log("[qspi] WARN: manufacturer != Winbond(0xEF)\r\n");
        return -2;
    }
    return 0;
}

static int qspi_erase_range(uint32_t flash_ofs, uint32_t size) {
    if ((flash_ofs % W25_BLOCK_SIZE) != 0)
        uart_log("[qspi] WARN erase offset 0x%lX not 64KB-aligned\r\n",
                 (unsigned long)flash_ofs);
    uint32_t end = flash_ofs + size;
    for (uint32_t a = flash_ofs & ~(W25_BLOCK_SIZE - 1); a < end; a += W25_BLOCK_SIZE) {
        if (qspi_write_enable()) return -1;
        QSPI_CommandTypeDef c = {0};
        c.InstructionMode = QSPI_INSTRUCTION_1_LINE;
        c.Instruction     = CMD_BLOCK_ERASE_64K;
        c.AddressMode     = QSPI_ADDRESS_1_LINE;
        c.AddressSize     = QSPI_ADDRESS_24_BITS;
        c.Address         = a;
        c.DataMode        = QSPI_DATA_NONE;
        if (HAL_QSPI_Command(&hqspi, &c, QSPI_TMO) != HAL_OK) return -2;
        if (qspi_wait_wip(1000)) return -3;  /* erase 64KB ~ vài trăm ms */
    }
    return 0;
}

static int qspi_program_page(uint32_t flash_addr, const uint8_t *data, uint32_t len) {
    if (qspi_write_enable()) return -1;
    QSPI_CommandTypeDef c = {0};
    c.InstructionMode = QSPI_INSTRUCTION_1_LINE;
    c.Instruction     = CMD_PAGE_PROGRAM;
    c.AddressMode     = QSPI_ADDRESS_1_LINE;
    c.AddressSize     = QSPI_ADDRESS_24_BITS;
    c.Address         = flash_addr;
    c.DataMode        = QSPI_DATA_1_LINE;
    c.NbData          = len;
    if (HAL_QSPI_Command(&hqspi, &c, QSPI_TMO) != HAL_OK) return -2;
    if (HAL_QSPI_Transmit(&hqspi, (uint8_t *)data, QSPI_TMO) != HAL_OK) return -3;
    return qspi_wait_wip(50);
}

int qspi_enable_memmap(void) {
    QSPI_CommandTypeDef c = {0};
    QSPI_MemoryMappedTypeDef m = {0};
    c.InstructionMode = QSPI_INSTRUCTION_1_LINE;
    c.Instruction     = CMD_FAST_READ;
    c.AddressMode     = QSPI_ADDRESS_1_LINE;
    c.AddressSize     = QSPI_ADDRESS_24_BITS;
    c.DataMode        = QSPI_DATA_1_LINE;
    c.DummyCycles     = 8;
    m.TimeOutActivation = QSPI_TIMEOUT_COUNTER_DISABLE;
    return HAL_QSPI_MemoryMapped(&hqspi, &c, &m) == HAL_OK ? 0 : -1;
}

int qspi_load_app_if_needed(const char *path) {
    HAL_QSPI_Abort(&hqspi);
    uint32_t head = 0;
    (void)qspi_read_data(0, (uint8_t *)&head, sizeof(head));

    if (sdcard_mount() != 0) return -1;

    /* Check flag file 0:/REPROGRAM — nếu có → force reprogram + xóa flag */
    FIL fflag;
    int force = (f_open(&fflag, "0:/REPROGRAM", FA_READ) == FR_OK);
    if (force) {
        f_close(&fflag);
        f_unlink("0:/REPROGRAM");
        uart_log("[qspi] REPROGRAM flag → force reload\r\n");
    }

    uart_log("[qspi] loading '%s'\r\n", path);
    FIL f;
    if (sdcard_open_read(&f, path) != FR_OK) {
        if (head != 0xFFFFFFFFu) {
            uart_log("[qspi] WARN: app file missing, keep existing slot head=0x%08lX\r\n",
                     (unsigned long)head);
            return 0;
        }
        return -2;
    }
    uint32_t size = (uint32_t)f_size(&f);
    if (!size || size > MODEL_SLOT_OFFSET) {
        uart_log("[qspi] bad app size %lu B\r\n", (unsigned long)size);
        f_close(&f); return -3;
    }

    uint32_t file_crc_val = 0;
    if (file_crc(&f, &file_crc_val) != 0) { f_close(&f); return -4; }
    if (!force) {
        uint32_t slot_crc = 0;
        if (qspi_crc_indirect(0, size, &slot_crc) == 0 && slot_crc == file_crc_val) {
            uart_log("[qspi] app slot OK size=%lu CRC=%08lX\r\n",
                     (unsigned long)size, (unsigned long)slot_crc);
            f_close(&f);
            return 0;
        }
        uart_log("[qspi] app slot stale/missing, reprogram CRC=%08lX\r\n",
                 (unsigned long)file_crc_val);
    }

    if (qspi_erase_range(0, size)) { f_close(&f); return -4; }
    uart_log("[qspi] programming %lu B...\r\n", (unsigned long)size);
    uint32_t addr = 0; UINT br;
    while (f_read(&f, s_io, sizeof(s_io), &br) == FR_OK && br) {
        for (uint32_t off = 0; off < br; off += W25_PAGE_SIZE) {
            uint32_t n = (br - off > W25_PAGE_SIZE) ? W25_PAGE_SIZE : br - off;
            if (qspi_program_page(addr + off, s_io + off, n)) { f_close(&f); return -5; }
        }
        addr += br;
    }
    f_close(&f);
    uart_log("[qspi] app loaded %lu B OK\r\n", (unsigned long)addr);
    return 0;
}

int qspi_load_model(const char *path, uint32_t flash_ofs,
                    uint32_t crc_expected, uint32_t *out_size) {
    FIL f;
    if (sdcard_open_read(&f, path) != FR_OK) return -1;
    uint32_t size = (uint32_t)f_size(&f);
    *out_size = size;

    if (size > MODEL_SLOT_SIZE) {
        uart_log("[qspi] model %lu B > slot %lu B\r\n",
                 (unsigned long)size, (unsigned long)MODEL_SLOT_SIZE);
        f_close(&f); return -2;
    }

    /* rời memmap trước khi erase/program */
    HAL_QSPI_Abort(&hqspi);

    uart_log("[qspi] erase %lu B @ofs 0x%lX ...\r\n",
             (unsigned long)size, (unsigned long)flash_ofs);
    if (qspi_erase_range(flash_ofs, size)) { f_close(&f); return -3; }

    uart_log("[qspi] program ...\r\n");
    uint32_t addr = flash_ofs;
    UINT br;
    while (1) {
        if (f_read(&f, s_io, sizeof(s_io), &br) != FR_OK) { f_close(&f); return -4; }
        if (br == 0) break;
        for (uint32_t off = 0; off < br; off += W25_PAGE_SIZE) {
            uint32_t n = br - off;
            if (n > W25_PAGE_SIZE) n = W25_PAGE_SIZE;
            if (qspi_program_page(addr + off, s_io + off, n)) { f_close(&f); return -5; }
        }
        addr += br;
    }
    f_close(&f);

    /* đọc lại qua memmap + CRC32 */
    if (qspi_enable_memmap()) return -6;
    board_qspi_cache_invalidate(QSPI_BASE_ADDR + flash_ofs, size);
    uint32_t crc = crc32_buf((const void *)(QSPI_BASE_ADDR + flash_ofs), size);
    if (crc != crc_expected) {
        uart_log("[qspi] CRC MISMATCH got=%08lX exp=%08lX\r\n",
                 (unsigned long)crc, (unsigned long)crc_expected);
        return -7;
    }
    uart_log("[qspi] loaded OK, CRC=%08lX\r\n", (unsigned long)crc);
    return 0;
}
