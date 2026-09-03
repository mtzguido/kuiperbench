
#include "Kuiper_KB_Matmul4D.h"

__global__
/**
  hoisted when extracting matmul4d_f32
*/
static void
__hoisted_matmul4d_f32_0(uint32_t l, uint32_t k, float *gA, float *gB,
                         float *gC, uint32_t nm)
{
    if (1024U * blockIdx.x + threadIdx.x < nm * k) {
        uint32_t trow = (1024U * blockIdx.x + threadIdx.x) / k;
        uint32_t tcol = (1024U * blockIdx.x + threadIdx.x) % k;
        uint32_t k1 = 0U;
        float acc = 0.0f;
        float c = 0.0f;
        for (; k1 < l; k1++) {
            uint32_t __anf0 = k1;
            float old_acc = acc;
            float yc = gA[trow * l + __anf0] * gB[__anf0 * k + tcol] - c;
            float t = old_acc + yc;
            c = t - old_acc - yc;
            acc = t;
        }
        gC[trow * k + tcol] = acc;
    }
}

void Kuiper_KB_Matmul4D_matmul4d_f32(uint32_t b, uint32_t i, uint32_t j,
                                     uint32_t l, uint32_t k, float *gA,
                                     float *gB, float *gC)
{
    KPR_GUARD(i <= 4294967295U / b);
    uint32_t bi = b * i;
    KPR_GUARD(j <= 4294967295U / bi);
    uint32_t bij = bi * j;
    KPR_GUARD(l <= 4294967295U / bij);
    KPR_GUARD(k <= 4294967295U / bij);
    uint32_t bijk = bij * k;
    KPR_GUARD(k <= 4294967295U / l);
    KPR_GUARD(bijk <= 2147483648U);
    uint32_t nm = bi * j;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_matmul4d_f32_0,
              nm * k / 1024U + (uint32_t) (nm * k % 1024U != 0U), 1024U, 0U, s,
              l, k, gA, gB, gC, nm);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

float *Kuiper_KB_Matmul4D_matmul4d_alloc_f32(uint32_t b, uint32_t i, uint32_t j,
                                             uint32_t l, uint32_t k, float *gA,
                                             float *gB)
{
    float *gC = (float *) KPR_GPU_ALLOC(sizeof(float), b * i * j * k);
    Kuiper_KB_Matmul4D_matmul4d_f32(b, i, j, l, k, gA, gB, gC);
    return gC;
}
