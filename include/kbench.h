#pragma once

#include <kuiper.h>
#include <cuda_runtime.h>

// KaRaMeL emits the five void* arguments as expected `(void *)0U` placeholders
// for erased Pulse sizes, permissions, and sequences.  The trusted shim ignores
// them by design; only the arrays, offsets, and count are runtime inputs.
template <typename T>
static inline void Kuiper_KB_Compat_Array_gpu_memcpy_device_to_device_(
    void *, T *dst, uint32_t dst_off, void *, T *src, uint32_t src_off,
    uint32_t cnt, void *, void *, void *)
{
    MUST(cudaMemcpy(dst + dst_off, src + src_off,
                    (size_t)cnt * sizeof(T), cudaMemcpyDeviceToDevice));
}
