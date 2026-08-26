
#include "Kuiper_Example_OffsetMemcpyD2D.h"

uint64_t Kuiper_Example_OffsetMemcpyD2D_main(void)
{
    uint64_t *src = (uint64_t *) KRML_HOST_CALLOC(8U, sizeof(uint64_t));
    *src = 10ULL;
    uint64_t *ga = (uint64_t *) KPR_GPU_ALLOC(sizeof(uint64_t), 8U);
    uint64_t *zeros = (uint64_t *) KRML_HOST_CALLOC(8U, sizeof(uint64_t));
    MUST(cudaMemcpy(ga, zeros, (uint32_t) sizeof(uint64_t) * 8U,
                    cudaMemcpyHostToDevice));
    KRML_HOST_FREE(zeros);
    MUST(cudaMemcpy(ga, src, (uint32_t) sizeof(uint64_t) * 8U,
                    cudaMemcpyHostToDevice));
    KRML_HOST_FREE(src);
    uint64_t *gb = (uint64_t *) KPR_GPU_ALLOC(sizeof(uint64_t), 8U);
    uint64_t *zb = (uint64_t *) KRML_HOST_CALLOC(8U, sizeof(uint64_t));
    MUST(cudaMemcpy(gb, zb, (uint32_t) sizeof(uint64_t) * 8U,
                    cudaMemcpyHostToDevice));
    KRML_HOST_FREE(zb);
    Kuiper_KB_Compat_Array_gpu_memcpy_device_to_device_(
        (void *) 0U, gb, 1U, (void *) 0U, ga, 2U, 3U, (void *) 0U, (void *) 0U,
        (void *) 0U);
    uint64_t *dst = (uint64_t *) KRML_HOST_CALLOC(8U, sizeof(uint64_t));
    MUST(cudaMemcpy(dst, gb, (uint32_t) sizeof(uint64_t) * 8U,
                    cudaMemcpyDeviceToHost));
    MUST(cudaFree(ga));
    MUST(cudaFree(gb));
    uint64_t r = dst[1U];
    KRML_HOST_FREE(dst);
    return r;
}
