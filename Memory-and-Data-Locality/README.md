# Memory and Data Locality

## Memory Access Efficiency

**compute-to-global-memory-access ratio** is ratio of floating-point calculation to global memory access operation. This ratio has major implications on performance of our kernel.

In `imageblur.cu` in **Scalable Parallel Executing**, the global memory fetches an in[] array element every iteration. The **compute-to-global-memory-access ratio** is 1 to 1, or `1.0`.

If our global memory bandwidth is 100GB/s, it can perform 250GFLOPS. When our kernel has 1.0 compute-to-global-memory-access ratio, the kernel will achieve no more than 250GFLOPS.

To maximize our kernel performance, we need to increase our ratio by reducing the number of global memory access.

## Matrix Multiplication

`mat_mul.cu` 