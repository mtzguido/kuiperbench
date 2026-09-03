
#include "Kuiper_KB_ConvT3DGrouped72.h"

__global__
/**
  hoisted when extracting convt3d_grouped72_alloc_f32
*/
static void
__hoisted_convt3d_grouped72_alloc_f32_0(float *gbias)
{
    if (1024U * blockIdx.x + threadIdx.x < 32U)
        gbias[1024U * blockIdx.x + threadIdx.x] = 0.0f;
}

__global__
/**
  hoisted when extracting convt3d_grouped72_alloc_f32
*/
static void
__hoisted_convt3d_grouped72_alloc_f32_1(float *gx, float *gw, float *gbias,
                                        float *gy)
{
    if (1024U * blockIdx.x + threadIdx.x < 28311552U) {
        uint32_t oc = (1024U * blockIdx.x + threadIdx.x) % 3538944U / 110592U;
        uint32_t g = oc / 8U;
        uint32_t oc_pg = oc % 8U;
        uint32_t r2 = (1024U * blockIdx.x + threadIdx.x) % 3538944U % 110592U;
        uint32_t r3 = r2 % 4608U;
        uint32_t od_pd = r2 / 4608U + 1U;
        uint32_t oh_ph = r3 / 96U + 2U;
        uint32_t ow_pw = r3 % 96U + 3U;
        float acc = 0.0f;
        uint32_t k = 0U;
        for (; k < 840U; k++) {
            uint32_t kk = k;
            uint32_t ic = g * 8U + kk / 105U;
            uint32_t r = kk % 105U;
            uint32_t kd_i = r / 35U;
            uint32_t r21 = r % 35U;
            uint32_t kh_i = r21 / 7U;
            uint32_t kw_i = r21 % 7U;
            uint32_t kd_dd = kd_i;
            uint32_t kh_dh = kh_i;
            uint32_t kw_dw = kw_i;
            float ite;
            if (od_pd >= kd_dd && oh_ph >= kh_dh && ow_pw >= kw_dw) {
                uint32_t d_num = od_pd - kd_dd;
                uint32_t h_num = oh_ph - kh_dh;
                uint32_t w_num = ow_pw - kw_dw;
                if (d_num % 2U == 0U && h_num % 2U == 0U && w_num % 2U == 0U) {
                    uint32_t di = d_num / 2U;
                    uint32_t hi = h_num / 2U;
                    uint32_t wi = w_num / 2U;
                    ite = di < 12U && hi < 24U && wi < 48U
                              ? gx[((((1024U * blockIdx.x + threadIdx.x) /
                                          3538944U * 32U +
                                      ic) *
                                         12U +
                                     di) *
                                        24U +
                                    hi) *
                                       48U +
                                   wi]
                              : 0.0f;
                } else
                    ite = 0.0f;
            } else
                ite = 0.0f;
            acc +=
                ite *
                gw[(((ic * 8U + oc_pg) * 3U + kd_i) * 5U + kh_i) * 7U + kw_i];
        }
        gy[1024U * blockIdx.x + threadIdx.x] = gbias[oc] + acc;
    }
}

float *Kuiper_KB_ConvT3DGrouped72_convt3d_grouped72_alloc_f32(float *gx,
                                                              float *gw)
{
    float *gbias = (float *) KPR_GPU_ALLOC(sizeof(float), 32U);
    cudaStream_t s0 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_convt3d_grouped72_alloc_f32_0, 1U, 1024U, 0U, s0,
              gbias);
    MUST(cudaStreamSynchronize(s0));
    MUST(cudaStreamDestroy(s0));
    float *gy = (float *) KPR_GPU_ALLOC(sizeof(float), 28311552U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_convt3d_grouped72_alloc_f32_1, 27648U, 1024U, 0U, s, gx,
              gw, gbias, gy);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    MUST(cudaFree(gbias));
    return gy;
}
