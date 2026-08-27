
#include "Kuiper_KB_Conv2DSquare.h"

uint32_t Kuiper_KB_Conv2DSquare_conv2d_square_out_sz(uint32_t l, uint32_t k)
{
    return l - k + 1U;
}

__global__
/**
  hoisted when extracting conv2d_square_f32
*/
static void
__hoisted_conv2d_square_f32_0(uint32_t b, uint32_t cin, uint32_t h_in,
                              uint32_t cout, uint32_t k, uint32_t h_out,
                              float *gx, float *gw, float *gbias, float *gy)
{
    if (1024U * blockIdx.x + threadIdx.x < b * cout * h_out * h_out) {
        uint32_t how = h_out * h_out;
        uint32_t chow = cout * how;
        uint32_t bi = (1024U * blockIdx.x + threadIdx.x) / chow;
        uint32_t r1 = (1024U * blockIdx.x + threadIdx.x) % chow;
        uint32_t oc = r1 / how;
        uint32_t r2 = r1 % how;
        uint32_t kh_kw = k * k;
        uint32_t n_taps = cin * kh_kw;
        uint32_t oh_s = r2 / h_out;
        uint32_t ow_s = r2 % h_out;
        float acc = 0.0f;
        uint32_t k1 = 0U;
        for (; k1 < n_taps; k1++) {
            uint32_t kk_v = k1;
            uint32_t ic = kk_v / kh_kw;
            uint32_t r = kk_v % kh_kw;
            uint32_t kh_i = r / k;
            uint32_t kw_i = r % k;
            uint32_t h_signed = oh_s + kh_i;
            uint32_t w_signed = ow_s + kw_i;
            float ite;
            if (0U <= h_signed && 0U <= w_signed) {
                uint32_t hi = h_signed - 0U;
                uint32_t wi = w_signed - 0U;
                ite = hi < h_in && wi < h_in
                          ? gx[((bi * cin + ic) * h_in + hi) * h_in + wi]
                          : 0.0f;
            } else
                ite = 0.0f;
            acc += ite * gw[((oc * cin + ic) * k + kh_i) * k + kw_i];
        }
        gy[1024U * blockIdx.x + threadIdx.x] = gbias[oc] + acc;
    }
}

void Kuiper_KB_Conv2DSquare_conv2d_square_f32(uint32_t b, uint32_t cin,
                                              uint32_t h_in, uint32_t cout,
                                              uint32_t k, uint32_t h_out,
                                              float *gx, float *gw,
                                              float *gbias, float *gy)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_conv2d_square_f32_0,
              b * cout * h_out * h_out / 1024U +
                  (uint32_t) (b * cout * h_out * h_out % 1024U != 0U),
              1024U, 0U, s, b, cin, h_in, cout, k, h_out, gx, gw, gbias, gy);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}
