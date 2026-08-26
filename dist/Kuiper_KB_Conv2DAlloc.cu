
#include "Kuiper_KB_Conv2DAlloc.h"

__global__
/**
  hoisted when extracting conv2d_general_f32
*/
static void
__hoisted_conv2d_general_f32_0(uint32_t b, uint32_t cin, uint32_t h_in,
                               uint32_t w_in, uint32_t cout, uint32_t kh,
                               uint32_t kw, uint32_t stride, uint32_t pad,
                               uint32_t h_out, uint32_t w_out, float *gx,
                               float *gw, float *gbias, float *gy)
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
        uint32_t oh_s = r2 / w_out * stride;
        uint32_t ow_s = r2 % w_out * stride;
        float acc = 0.0f;
        uint32_t k = 0U;
        for (; k < n_taps; k = kk_v + 1U) {
            uint32_t kk_v = k;
            uint32_t ic = kk_v / kh_kw;
            uint32_t r = kk_v % kh_kw;
            uint32_t kh_i = r / kw;
            uint32_t kw_i = r % kw;
            uint32_t h_signed = oh_s + kh_i;
            uint32_t w_signed = ow_s + kw_i;
            float ite;
            if (pad <= h_signed && pad <= w_signed) {
                uint32_t hi = h_signed - pad;
                uint32_t wi = w_signed - pad;
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

static void conv2d_general_f32(uint32_t b, uint32_t cin, uint32_t h_in,
                               uint32_t w_in, uint32_t cout, uint32_t kh,
                               uint32_t kw, uint32_t stride, uint32_t pad,
                               uint32_t h_out, uint32_t w_out, float *gx,
                               float *gw, float *gbias, float *gy)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_conv2d_general_f32_0,
              b * cout * h_out * w_out / 1024U +
                  (uint32_t) (b * cout * h_out * w_out % 1024U != 0U),
              1024U, 0U, s, b, cin, h_in, w_in, cout, kh, kw, stride, pad,
              h_out, w_out, gx, gw, gbias, gy);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

uint32_t Kuiper_KB_Conv2DAlloc_conv2d_out_dim(uint32_t n, uint32_t k,
                                              uint32_t stride, uint32_t pad)
{
    return (n + 2U * pad - k) / stride + 1U;
}

float *Kuiper_KB_Conv2DAlloc_conv2d_general_alloc_f32(
    uint32_t b, uint32_t cin, uint32_t h_in, uint32_t w_in, uint32_t cout,
    uint32_t kh, uint32_t kw, uint32_t stride, uint32_t pad, uint32_t h_out,
    uint32_t w_out, float *gx, float *gw, float *gbias)
{
    float *gy =
        (float *) KPR_GPU_ALLOC(sizeof(float), b * cout * h_out * w_out);
    conv2d_general_f32(b, cin, h_in, w_in, cout, kh, kw, stride, pad, h_out,
                       w_out, gx, gw, gbias, gy);
    return gy;
}
