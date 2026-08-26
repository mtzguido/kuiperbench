
#include "Kuiper_KB_Conv2DDilatedAsym.h"

uint32_t Kuiper_KB_Conv2DDilatedAsym_conv2dd_out_dim_sz(uint32_t l, uint32_t k,
                                                        uint32_t s, uint32_t d,
                                                        uint32_t p)
{
    uint32_t kspan = d * (k - 1U) + 1U;
    uint32_t padded = l + 2U * p;
    if (padded < kspan)
        return 0U;
    else
        return (padded - kspan) / s + 1U;
}

__global__
/**
  hoisted when extracting conv2d_dilated_asym_f32
*/
static void
__hoisted_conv2d_dilated_asym_f32_0(uint32_t b, uint32_t cin, uint32_t h_in,
                                    uint32_t w_in, uint32_t cout, uint32_t kh,
                                    uint32_t kw, uint32_t sh, uint32_t sw,
                                    uint32_t ph, uint32_t pw, uint32_t dh,
                                    uint32_t dw, uint32_t h_out, uint32_t w_out,
                                    float *gx, float *gw, float *gbias,
                                    float *gy)
{
    if (1024U * blockIdx.x + threadIdx.x < b * cout * h_out * w_out) {
        uint32_t how = h_out * w_out;
        uint32_t chow = cout * how;
        uint32_t bi = (1024U * blockIdx.x + threadIdx.x) / chow;
        uint32_t r1 = (1024U * blockIdx.x + threadIdx.x) % chow;
        uint32_t oc = r1 / how;
        uint32_t r2 = r1 % how;
        uint32_t kh_kw = kh * kw;
        uint32_t n_taps = cin * kh_kw;
        uint32_t oh_s = r2 / w_out * sh;
        uint32_t ow_s = r2 % w_out * sw;
        float acc = 0.0f;
        uint32_t k = 0U;
        for (; k < n_taps; k = kk_v + 1U) {
            uint32_t kk_v = k;
            uint32_t ic = kk_v / kh_kw;
            uint32_t r = kk_v % kh_kw;
            uint32_t kh_i = r / kw;
            uint32_t kw_i = r % kw;
            uint32_t h_signed = oh_s + kh_i * dh;
            uint32_t w_signed = ow_s + kw_i * dw;
            float ite;
            if (ph <= h_signed && pw <= w_signed) {
                uint32_t hi = h_signed - ph;
                uint32_t wi = w_signed - pw;
                ite = hi < h_in && wi < w_in
                          ? gx[((bi * cin + ic) * h_in + hi) * w_in + wi]
                          : 0.0f;
            } else
                ite = 0.0f;
            acc += ite * gw[((oc * cin + ic) * kh + kh_i) * kw + kw_i];
        }
        gy[1024U * blockIdx.x + threadIdx.x] = gbias[oc] + acc;
    }
}

void Kuiper_KB_Conv2DDilatedAsym_conv2d_dilated_asym_f32(
    uint32_t b, uint32_t cin, uint32_t h_in, uint32_t w_in, uint32_t cout,
    uint32_t kh, uint32_t kw, uint32_t sh, uint32_t sw, uint32_t ph,
    uint32_t pw, uint32_t dh, uint32_t dw, uint32_t h_out, uint32_t w_out,
    float *gx, float *gw, float *gbias, float *gy)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_conv2d_dilated_asym_f32_0,
              b * cout * h_out * w_out / 1024U +
                  (uint32_t) (b * cout * h_out * w_out % 1024U != 0U),
              1024U, 0U, s, b, cin, h_in, w_in, cout, kh, kw, sh, sw, ph, pw,
              dh, dw, h_out, w_out, gx, gw, gbias, gy);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}
