
#include "Kuiper_KB_BatchedGEMM.h"

__global__
/**
  hoisted when extracting batched_gemm_f32
*/
static void
__hoisted_batched_gemm_f32_0(uint32_t batch, uint32_t rows, uint32_t shared,
                             uint32_t cols, float *a, float *b, float *c)
{
    if (1024U * blockIdx.x + threadIdx.x < batch * rows * cols) {
        uint32_t page = (1024U * blockIdx.x + threadIdx.x) % batch;
        uint32_t rest = (1024U * blockIdx.x + threadIdx.x) / batch;
        uint32_t trow = rest / cols;
        uint32_t tcol = rest % cols;
        uint32_t k = 0U;
        float sum = 0.0f;
        for (; k < shared; k++) {
            uint32_t vk = k;
            sum += a[page * rows * shared + trow * shared + vk] *
                   b[page * shared * cols + vk * cols + tcol];
        }
        c[page * rows * cols + trow * cols + tcol] = sum;
    }
}

void Kuiper_KB_BatchedGEMM_batched_gemm_f32(uint32_t batch, uint32_t rows,
                                            uint32_t shared, uint32_t cols,
                                            float *a, float *b, float *c)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_batched_gemm_f32_0,
              batch * rows * cols / 1024U +
                  (uint32_t) (batch * rows * cols % 1024U != 0U),
              1024U, 0U, s, batch, rows, shared, cols, a, b, c);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}
