
#include "Kuiper_KB_ConvT2DGrouped75.h"

__global__
/**
  hoisted when extracting convt2d_grouped75_alloc_f32
*/
static void
__hoisted_convt2d_grouped75_alloc_f32_0(float *gbias)
{
    if (1024U * blockIdx.x + threadIdx.x < 64U)
        gbias[1024U * blockIdx.x + threadIdx.x] = 0.0f;
}

__global__
/**
  hoisted when extracting convt2d_grouped75_alloc_f32
*/
static void
__hoisted_convt2d_grouped75_alloc_f32_1(float *gx, float *gw, float *gbias,
                                        float *gy)
{
    if (1024U * blockIdx.x + threadIdx.x < 201586688U) {
        uint32_t g =
            (1024U * blockIdx.x + threadIdx.x) % 12599168U / 196862U / 16U;
        uint32_t oc_pg =
            (1024U * blockIdx.x + threadIdx.x) % 12599168U / 196862U % 16U;
        uint32_t oh_ph =
            (1024U * blockIdx.x + threadIdx.x) % 12599168U % 196862U / 766U +
            1U;
        uint32_t ow_pw =
            (1024U * blockIdx.x + threadIdx.x) % 12599168U % 196862U % 766U +
            2U;
        float acc = 0.0f;
        uint32_t k = 0U;
        for (; k < 120U; k++) {
            uint32_t kk = k;
            uint32_t ic = g * 8U + kk / 15U;
            uint32_t r = kk % 15U;
            uint32_t kh_i = r / 5U;
            uint32_t kw_i = r % 5U;
            uint32_t kh_dh = kh_i * 2U;
            uint32_t kw_dw = kw_i;
            float ite;
            if (oh_ph >= kh_dh && ow_pw >= kw_dw) {
                uint32_t h_num = oh_ph - kh_dh;
                uint32_t w_num = ow_pw - kw_dw;
                if (h_num % 2U == 0U && w_num % 3U == 0U) {
                    uint32_t hi = h_num / 2U;
                    uint32_t wi = w_num / 3U;
                    ite = hi < 128U && wi < 256U
                              ? gx[(((1024U * blockIdx.x + threadIdx.x) /
                                         12599168U * 32U +
                                     ic) *
                                        128U +
                                    hi) *
                                       256U +
                                   wi]
                              : 0.0f;
                } else
                    ite = 0.0f;
            } else
                ite = 0.0f;
            acc += ite * gw[((ic * 16U + oc_pg) * 3U + kh_i) * 5U + kw_i];
        }
        gy[1024U * blockIdx.x + threadIdx.x] =
            gbias[(1024U * blockIdx.x + threadIdx.x) % 12599168U / 196862U] +
            acc;
    }
}

float *Kuiper_KB_ConvT2DGrouped75_convt2d_grouped75_alloc_f32(float *gx,
                                                              float *gw)
{
    float *gbias = (float *) KPR_GPU_ALLOC(sizeof(float), 64U);
    cudaStream_t s0 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_convt2d_grouped75_alloc_f32_0, 1U, 1024U, 0U, s0,
              gbias);
    MUST(cudaStreamSynchronize(s0));
    MUST(cudaStreamDestroy(s0));
    float *gy = (float *) KPR_GPU_ALLOC(sizeof(float), 201586688U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_convt2d_grouped75_alloc_f32_1, 196862U, 1024U, 0U, s,
              gx, gw, gbias, gy);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    MUST(cudaFree(gbias));
    return gy;
}
