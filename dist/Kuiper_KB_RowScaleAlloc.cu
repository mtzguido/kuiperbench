
#include "Kuiper_KB_RowScaleAlloc.h"

float *Kuiper_KB_RowScaleAlloc_row_scale_alloc_f32(uint32_t m, uint32_t n,
                                                   float *a, float *b)
{
    float *out = (float *) KPR_GPU_ALLOC(sizeof(float), m * n);
    MUST(cudaMemcpy(out, b, (uint32_t) sizeof(float) * m * n,
                    cudaMemcpyDeviceToDevice));
    Klas_RowScale_rowscale_f32_rowmajor(m, n, (void *) 0U, a, out, (void *) 0U,
                                        (void *) 0U, (void *) 0U, (void *) 0U,
                                        (void *) 0U);
    return out;
}
