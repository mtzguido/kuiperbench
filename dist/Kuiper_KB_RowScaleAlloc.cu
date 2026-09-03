
#include "Kuiper_KB_RowScaleAlloc.h"

__global__
/**
  hoisted when extracting rowscale_f32_rowmajor
*/
static void
__hoisted_rowscale_f32_rowmajor_0(uint32_t m, uint32_t n, float *a, float *b)
{
    if (1024U * blockIdx.x + threadIdx.x < m * n) {
        uint32_t row = (1024U * blockIdx.x + threadIdx.x) / n;
        uint32_t col = (1024U * blockIdx.x + threadIdx.x) % n;
        b[row * n + col] *= a[row];
    }
}

static void rowscale_f32_rowmajor(uint32_t m, uint32_t n, float *a, float *b)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_rowscale_f32_rowmajor_0,
              m * n / 1024U + (uint32_t) (m * n % 1024U != 0U), 1024U, 0U, s, m,
              n, a, b);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

float *Kuiper_KB_RowScaleAlloc_row_scale_alloc_f32(uint32_t m, uint32_t n,
                                                   float *a, float *b)
{
    float *out = (float *) KPR_GPU_ALLOC(sizeof(float), m * n);
    MUST(cudaMemcpy(out, b, (uint32_t) sizeof(float) * m * n,
                    cudaMemcpyDeviceToDevice));
    rowscale_f32_rowmajor(m, n, a, out);
    return out;
}
