
#include "Kuiper_KB_Conv1DGeneral.h"

__global__
/**
  hoisted when extracting conv1d_general_f32
*/
static void
__hoisted_conv1d_general_f32_0(uint32_t b, uint32_t cin, uint32_t l_in,
                               uint32_t cout, uint32_t kk, uint32_t stride,
                               uint32_t pad, uint32_t dilation, uint32_t l_out,
                               float *gx, float *gw, float *gbias, float *gy)
{
    if (1024U * blockIdx.x + threadIdx.x < b * cout * l_out) {
        uint32_t cl = cout * l_out;
        uint32_t bi = (1024U * blockIdx.x + threadIdx.x) / cl;
        uint32_t r1 = (1024U * blockIdx.x + threadIdx.x) % cl;
        uint32_t oc = r1 / l_out;
        uint32_t n_taps = cin * kk;
        uint32_t ol_s = r1 % l_out * stride;
        float acc = 0.0f;
        uint32_t k = 0U;
        for (; k < n_taps; k++) {
            uint32_t kk_v = k;
            uint32_t ic = kk_v / kk;
            uint32_t k_i = kk_v % kk;
            uint32_t l_signed = ol_s + k_i * dilation;
            float ite;
            if (pad <= l_signed) {
                uint32_t li = l_signed - pad;
                ite = li < l_in ? gx[(bi * cin + ic) * l_in + li] : 0.0f;
            } else
                ite = 0.0f;
            acc += ite * gw[(oc * cin + ic) * kk + k_i];
        }
        gy[1024U * blockIdx.x + threadIdx.x] = gbias[oc] + acc;
    }
}

void Kuiper_KB_Conv1DGeneral_conv1d_general_f32(
    uint32_t b, uint32_t cin, uint32_t l_in, uint32_t cout, uint32_t kk,
    uint32_t stride, uint32_t pad, uint32_t dilation, uint32_t l_out, float *gx,
    float *gw, float *gbias, float *gy)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_conv1d_general_f32_0,
              b * cout * l_out / 1024U +
                  (uint32_t) (b * cout * l_out % 1024U != 0U),
              1024U, 0U, s, b, cin, l_in, cout, kk, stride, pad, dilation,
              l_out, gx, gw, gbias, gy);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}
