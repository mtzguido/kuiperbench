
#include "Kuiper_KB_DepthwiseConv2D.h"

uint32_t Kuiper_KB_DepthwiseConv2D_dwconv2d_out_dim(uint32_t n, uint32_t k,
                                                    uint32_t stride,
                                                    uint32_t pad)
{
    return (n + 2U * pad - k) / stride + 1U;
}

__global__
/**
  hoisted when extracting dwconv2d_f32
*/
static void
__hoisted_dwconv2d_f32_0(uint32_t b, uint32_t c, uint32_t h_in, uint32_t w_in,
                         uint32_t kh, uint32_t kw, uint32_t stride,
                         uint32_t pad, uint32_t h_out, uint32_t w_out,
                         float *gx, float *gw, float *gbias, float *gy)
{
    if (1024U * blockIdx.x + threadIdx.x < b * c * h_out * w_out) {
        uint32_t how = h_out * w_out;
        uint32_t chow = c * how;
        uint32_t bi = (1024U * blockIdx.x + threadIdx.x) / chow;
        uint32_t r1 = (1024U * blockIdx.x + threadIdx.x) % chow;
        uint32_t ci = r1 / how;
        uint32_t r2 = r1 % how;
        uint32_t n_taps = kh * kw;
        uint32_t oh_s = r2 / w_out * stride;
        uint32_t ow_s = r2 % w_out * stride;
        float acc = 0.0f;
        uint32_t k = 0U;
        for (; k < n_taps; k++) {
            uint32_t kk_v = k;
            uint32_t kh_i = kk_v / kw;
            uint32_t kw_i = kk_v % kw;
            uint32_t h_signed = oh_s + kh_i;
            uint32_t w_signed = ow_s + kw_i;
            float ite;
            if (pad <= h_signed && pad <= w_signed) {
                uint32_t hi = h_signed - pad;
                uint32_t wi = w_signed - pad;
                ite = hi < h_in && wi < w_in
                          ? gx[((bi * c + ci) * h_in + hi) * w_in + wi]
                          : 0.0f;
            } else
                ite = 0.0f;
            acc += ite * gw[(ci * kh + kh_i) * kw + kw_i];
        }
        gy[1024U * blockIdx.x + threadIdx.x] = gbias[ci] + acc;
    }
}

void Kuiper_KB_DepthwiseConv2D_dwconv2d_f32(
    uint32_t b, uint32_t c, uint32_t h_in, uint32_t w_in, uint32_t kh,
    uint32_t kw, uint32_t stride, uint32_t pad, uint32_t h_out, uint32_t w_out,
    float *gx, float *gw, float *gbias, float *gy)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_dwconv2d_f32_0,
              b * c * h_out * w_out / 1024U +
                  (uint32_t) (b * c * h_out * w_out % 1024U != 0U),
              1024U, 0U, s, b, c, h_in, w_in, kh, kw, stride, pad, h_out, w_out,
              gx, gw, gbias, gy);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting dwconv2d_alloc_f32
*/
static void
__hoisted_dwconv2d_alloc_f32_0(uint32_t b, uint32_t c, uint32_t h_in,
                               uint32_t w_in, uint32_t kh, uint32_t kw,
                               uint32_t stride, uint32_t pad, uint32_t h_out,
                               uint32_t w_out, float *gx, float *gw,
                               float *gbias, float *gy)
{
    if (1024U * blockIdx.x + threadIdx.x < b * c * h_out * w_out) {
        uint32_t how = h_out * w_out;
        uint32_t chow = c * how;
        uint32_t bi = (1024U * blockIdx.x + threadIdx.x) / chow;
        uint32_t r1 = (1024U * blockIdx.x + threadIdx.x) % chow;
        uint32_t ci = r1 / how;
        uint32_t r2 = r1 % how;
        uint32_t n_taps = kh * kw;
        uint32_t oh_s = r2 / w_out * stride;
        uint32_t ow_s = r2 % w_out * stride;
        float acc = 0.0f;
        uint32_t k = 0U;
        for (; k < n_taps; k++) {
            uint32_t kk_v = k;
            uint32_t kh_i = kk_v / kw;
            uint32_t kw_i = kk_v % kw;
            uint32_t h_signed = oh_s + kh_i;
            uint32_t w_signed = ow_s + kw_i;
            float ite;
            if (pad <= h_signed && pad <= w_signed) {
                uint32_t hi = h_signed - pad;
                uint32_t wi = w_signed - pad;
                ite = hi < h_in && wi < w_in
                          ? gx[((bi * c + ci) * h_in + hi) * w_in + wi]
                          : 0.0f;
            } else
                ite = 0.0f;
            acc += ite * gw[(ci * kh + kh_i) * kw + kw_i];
        }
        gy[1024U * blockIdx.x + threadIdx.x] = gbias[ci] + acc;
    }
}

float *Kuiper_KB_DepthwiseConv2D_dwconv2d_alloc_f32(
    uint32_t b, uint32_t c, uint32_t h_in, uint32_t w_in, uint32_t kh,
    uint32_t kw, uint32_t stride, uint32_t pad, uint32_t h_out, uint32_t w_out,
    float *gx, float *gw, float *gbias)
{
    float *gy = (float *) KPR_GPU_ALLOC(sizeof(float), b * c * h_out * w_out);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_dwconv2d_alloc_f32_0,
              b * c * h_out * w_out / 1024U +
                  (uint32_t) (b * c * h_out * w_out % 1024U != 0U),
              1024U, 0U, s, b, c, h_in, w_in, kh, kw, stride, pad, h_out, w_out,
              gx, gw, gbias, gy);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    return gy;
}
