
#include "Kuiper_KB_Conv3DGeneral.h"

__global__
/**
  hoisted when extracting conv3d_general_f32
*/
static void
__hoisted_conv3d_general_f32_0(uint32_t b, uint32_t cin, uint32_t d_in,
                               uint32_t h_in, uint32_t w_in, uint32_t cout,
                               uint32_t kd, uint32_t kh, uint32_t kw,
                               uint32_t stride, uint32_t pad, uint32_t d_out,
                               uint32_t h_out, uint32_t w_out, float *gx,
                               float *gw, float *gbias, float *gy)
{
    if (1024U * blockIdx.x + threadIdx.x < b * cout * d_out * h_out * w_out) {
        uint32_t how = h_out * w_out;
        uint32_t dhw = d_out * how;
        uint32_t cdhw = cout * dhw;
        uint32_t bi = (1024U * blockIdx.x + threadIdx.x) / cdhw;
        uint32_t r1 = (1024U * blockIdx.x + threadIdx.x) % cdhw;
        uint32_t oc = r1 / dhw;
        uint32_t r2 = r1 % dhw;
        uint32_t r3 = r2 % how;
        uint32_t kh_kw = kh * kw;
        uint32_t kd_kh_kw = kd * kh_kw;
        uint32_t n_taps = cin * kd_kh_kw;
        uint32_t od_s = r2 / how * stride;
        uint32_t oh_s = r3 / w_out * stride;
        uint32_t ow_s = r3 % w_out * stride;
        float acc = 0.0f;
        uint32_t k = 0U;
        for (; k < n_taps; k++) {
            uint32_t kk = k;
            uint32_t ic = kk / kd_kh_kw;
            uint32_t r = kk % kd_kh_kw;
            uint32_t kd_i = r / kh_kw;
            uint32_t r21 = r % kh_kw;
            uint32_t kh_i = r21 / kw;
            uint32_t kw_i = r21 % kw;
            uint32_t d_signed = od_s + kd_i;
            uint32_t h_signed = oh_s + kh_i;
            uint32_t w_signed = ow_s + kw_i;
            float ite;
            if (pad <= d_signed && pad <= h_signed && pad <= w_signed) {
                uint32_t di = d_signed - pad;
                uint32_t hi = h_signed - pad;
                uint32_t wi = w_signed - pad;
                ite = di < d_in && hi < h_in && wi < w_in
                          ? gx[(((bi * cin + ic) * d_in + di) * h_in + hi) *
                                   w_in +
                               wi]
                          : 0.0f;
            } else
                ite = 0.0f;
            acc += ite *
                   gw[(((oc * cin + ic) * kd + kd_i) * kh + kh_i) * kw + kw_i];
        }
        gy[1024U * blockIdx.x + threadIdx.x] = gbias[oc] + acc;
    }
}

void Kuiper_KB_Conv3DGeneral_conv3d_general_f32(
    uint32_t b, uint32_t cin, uint32_t d_in, uint32_t h_in, uint32_t w_in,
    uint32_t cout, uint32_t kd, uint32_t kh, uint32_t kw, uint32_t stride,
    uint32_t pad, uint32_t d_out, uint32_t h_out, uint32_t w_out, float *gx,
    float *gw, float *gbias, float *gy)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_conv3d_general_f32_0,
              b * cout * d_out * h_out * w_out / 1024U +
                  (uint32_t) (b * cout * d_out * h_out * w_out % 1024U != 0U),
              1024U, 0U, s, b, cin, d_in, h_in, w_in, cout, kd, kh, kw, stride,
              pad, d_out, h_out, w_out, gx, gw, gbias, gy);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}
