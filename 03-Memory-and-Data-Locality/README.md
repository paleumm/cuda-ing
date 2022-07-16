# Memory and Data Locality

## Memory Access Efficiency

**compute-to-global-memory-access ratio** is ratio of floating-point calculation to global memory access operation. This ratio has major implications on performance of our kernel.

In `imageblur.cu` in **Scalable Parallel Executing**, the global memory fetches an in[] array element every iteration. The **compute-to-global-memory-access ratio** is 1 to 1, or `1.0`.

If our global memory bandwidth is 100GB/s, it can perform 250GFLOPS. When our kernel has 1.0 compute-to-global-memory-access ratio, the kernel will achieve no more than 250GFLOPS.

To maximize our kernel performance, we need to increase our ratio by reducing the number of global memory access.

## CUDA Memory Types

> For **host** can access (R/W) `Global Memory` and `Constant Memory`

- `Global Memory` can be read and written by `device`.
- `Constant Memory` is **Read-Only** access by `device`, its has short-latency and high-bandwidth. `__device__ __constant__`
- `Registers Memory` (high-speed, higly parallel) are allocated to individual threads, thread cannot access other's register.
- `Shared Memory` (high-speed, higly parallel) are allocated to thread blocks. Can access throughout each thread block. `__shared__`


## Matrix Multiplication

`mat_mul.cu` 