/* STM32H750VBTx — XIP từ QSPI Flash (Bước B).
 * Bootstrap nhỏ + HAL + app kernel ở INTERNAL FLASH 128K (0x08000000).
 * TFLM + inference backend chạy XIP từ QSPI 0x90000000 (2MB app slot).
 * Model slot giữ nguyên 0x90200000 (6MB). */

ENTRY(Reset_Handler)

_estack = ORIGIN(DTCMRAM) + LENGTH(DTCMRAM);   /* đỉnh stack = cuối DTCM 128K */
_Min_Heap_Size  = 0x400;    /* 1 KB */
_Min_Stack_Size = 0x2000;   /* 8 KB */

MEMORY
{
  ITCMRAM   (xrw) : ORIGIN = 0x00000000, LENGTH = 64K
  FLASH     (rx)  : ORIGIN = 0x08000000, LENGTH = 128K
  QSPI_APP  (rx)  : ORIGIN = 0x90000000, LENGTH = 2M      /* TFLM XIP slot */
  DTCMRAM   (xrw) : ORIGIN = 0x20000000, LENGTH = 128K
  RAM_D1    (xrw) : ORIGIN = 0x24000000, LENGTH = 512K
  RAM_D2    (xrw) : ORIGIN = 0x30000000, LENGTH = 288K
  RAM_D3    (xrw) : ORIGIN = 0x38000000, LENGTH = 64K
}

SECTIONS
{
  /* === INTERNAL FLASH 128K: vector table + bootstrap + HAL + app kernel === */

  .isr_vector :
  {
    . = ALIGN(4);
    KEEP(*(.isr_vector))
    . = ALIGN(4);
  } >FLASH

  .text :
  {
    . = ALIGN(4);
    /* EXCLUDE_FILE: TFLM archive đi vào .qspi_text bên dưới, không vào đây */
    *(EXCLUDE_FILE(*libtflm_xip.a) .text)
    *(EXCLUDE_FILE(*libtflm_xip.a) .text*)
    *(.glue_7)
    *(.glue_7t)
    *(.eh_frame)
    KEEP (*(.init))
    KEEP (*(.fini))
    . = ALIGN(4);
    _etext = .;
  } >FLASH

  .rodata :
  {
    . = ALIGN(4);
    *(EXCLUDE_FILE(*libtflm_xip.a) .rodata)
    *(EXCLUDE_FILE(*libtflm_xip.a) .rodata*)
    . = ALIGN(4);
  } >FLASH

  .ARM.extab : { *(.ARM.extab* .gnu.linkonce.armextab.*) } >FLASH
  .ARM : {
    __exidx_start = .;
    *(.ARM.exidx*)
    __exidx_end = .;
  } >FLASH

  .preinit_array :
  {
    PROVIDE_HIDDEN (__preinit_array_start = .);
    KEEP (*(.preinit_array*))
    PROVIDE_HIDDEN (__preinit_array_end = .);
  } >FLASH
  .init_array :
  {
    PROVIDE_HIDDEN (__init_array_start = .);
    KEEP (*(SORT(.init_array.*)))
    KEEP (*(.init_array*))
    PROVIDE_HIDDEN (__init_array_end = .);
  } >FLASH
  .fini_array :
  {
    PROVIDE_HIDDEN (__fini_array_start = .);
    KEEP (*(SORT(.fini_array.*)))
    KEEP (*(.fini_array*))
    PROVIDE_HIDDEN (__fini_array_end = .);
  } >FLASH

  /* === QSPI XIP: TFLM + inference backend === */

  .qspi_text :
  {
    . = ALIGN(4);
    /* CMake tạo archive libtflm_xip.a — match bằng tên archive */
    *libtflm_xip.a:*(.text .text.*)
    . = ALIGN(4);
  } >QSPI_APP

  .qspi_rodata :
  {
    . = ALIGN(4);
    *libtflm_xip.a:*(.rodata .rodata.*)
    . = ALIGN(4);
  } >QSPI_APP

  /* === RAM: .data + .bss ở DTCM, arena ở RAM_D1 === */

  _sidata = LOADADDR(.data);
  .data :
  {
    . = ALIGN(4);
    _sdata = .;
    *(.data)
    *(.data*)
    . = ALIGN(4);
    _edata = .;
  } >DTCMRAM AT> FLASH

  . = ALIGN(4);
  .bss :
  {
    _sbss = .;
    __bss_start__ = _sbss;
    *(.bss)
    *(.bss*)
    *(COMMON)
    . = ALIGN(4);
    _ebss = .;
    __bss_end__ = _ebss;
  } >DTCMRAM

  ._user_heap_stack :
  {
    . = ALIGN(8);
    PROVIDE ( end = . );
    PROVIDE ( _end = . );
    . = . + _Min_Heap_Size;
    . = . + _Min_Stack_Size;
    . = ALIGN(8);
  } >DTCMRAM

  /* Tensor arena TFLM chiếm TRỌN RAM_D1 512K. */
  .tflm_arena (NOLOAD) :
  {
    . = ALIGN(16);
    *(.tflm_arena)
    . = ALIGN(16);
  } >RAM_D1

  /DISCARD/ : { libc.a ( * ) libm.a ( * ) libgcc.a ( * ) }
  .ARM.attributes 0 : { *(.ARM.attributes) }
}
