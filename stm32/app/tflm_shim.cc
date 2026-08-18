/* tflm_shim.cc — lớp đệm để TFLM (C++) chạy trên bare-metal H750 (nano libc, nosys).
 *
 *  1) DebugLog/DebugVsnprintf: TFLM gọi để in lỗi (arena/op) -> route về uart_log
 *     (đi vào cả RAM mailbox JLOG đọc qua ST-Link lẫn UART). RẤT hữu ích khi bring-up.
 *  2) C++ runtime tối thiểu: operator new/delete + __cxa stub, KHÔNG link libstdc++.
 *     new dùng pool tĩnh vì nosys.specs không có _sbrk (malloc fail). TFLM static-memory
 *     hầu như không new lúc chạy; pool chỉ là lưới an toàn + bắt lỗi.
 */
#include <cstdarg>
#include <cstddef>
#include <cstdio>

#include "board.h"   /* uart_log (extern "C") */
#include "tensorflow/lite/micro/debug_log.h"   /* prototype DebugLog/DebugVsnprintf */

extern "C" void DebugLog(const char *format, va_list args) {
    char buf[256];
    DebugVsnprintf(buf, sizeof(buf), format, args);
    uart_log("%s", buf);
}

extern "C" int DebugVsnprintf(char *buffer, size_t buf_size,
                              const char *format, va_list vlist) {
    return vsnprintf(buffer, buf_size, format, vlist);
}

/* ---- operator new/delete: pool tĩnh (bump) ---- */
static unsigned char s_cxx_pool[8192];
static size_t s_cxx_top = 0;

static void *cxx_alloc(size_t n) {
    n = (n + 7u) & ~static_cast<size_t>(7u);
    if (s_cxx_top + n > sizeof(s_cxx_pool)) {
        uart_log("[tflm] FATAL operator new pool OOM (+%u, used %u/%u)\r\n",
                 (unsigned)n, (unsigned)s_cxx_top, (unsigned)sizeof(s_cxx_pool));
        for (;;) {}
    }
    void *p = &s_cxx_pool[s_cxx_top];
    s_cxx_top += n;
    return p;
}

void *operator new(size_t n)            { return cxx_alloc(n); }
void *operator new[](size_t n)          { return cxx_alloc(n); }
void  operator delete(void *) noexcept   {}
void  operator delete[](void *) noexcept {}
void  operator delete(void *, size_t) noexcept   {}
void  operator delete[](void *, size_t) noexcept {}

/* ---- __cxa stub (no exceptions/rtti, -fno-threadsafe-statics) ---- */
extern "C" void __cxa_pure_virtual(void) { for (;;) {} }
extern "C" int  __cxa_atexit(void (*)(void *), void *, void *) { return 0; }
extern "C" int  __aeabi_atexit(void *, void (*)(void *), void *) { return 0; }
/* __dso_handle do crtbegin.o cấp — KHÔNG tự định nghĩa (tránh multiple definition). */
