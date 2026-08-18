/* kernel.h — điểm vào harness. Gọi từ main.c sau khi CubeMX init xong peripheral. */
#ifndef KERNEL_H
#define KERNEL_H

/* Vòng lặp manifest → mỗi model: nạp QSPI → infer toàn test set → metrics → results.
 * Không bao giờ return (halt sau khi xong). */
void kernel_main(void);

#endif /* KERNEL_H */
