
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
        for (; k < n_taps; k++) {
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

Kuiper_KB_Conv2DAlloc_conv2d_raw_result
Kuiper_KB_Conv2DAlloc_conv2d_raw_alloc_bias_f32(uint32_t b, uint32_t cin,
                                                uint32_t h_in, uint32_t w_in,
                                                uint32_t cout, uint32_t kh,
                                                uint32_t kw, uint32_t stride,
                                                uint32_t pad, float *gx,
                                                float *gw, float *gbias)
{
    KPR_GUARD(pad <= 2147483647U);
    uint32_t two_pad = 2U * pad;
    KPR_GUARD(h_in <= 4294967295U - two_pad);
    uint32_t h_pad = h_in + two_pad;
    KPR_GUARD(w_in <= 4294967295U - two_pad);
    uint32_t w_pad = w_in + two_pad;
    KPR_GUARD(kh <= h_pad);
    KPR_GUARD(kw <= w_pad);
    uint32_t h_out =
        Kuiper_KB_Conv2DAlloc_conv2d_out_dim(h_in, kh, stride, pad);
    uint32_t w_out0 =
        Kuiper_KB_Conv2DAlloc_conv2d_out_dim(w_in, kw, stride, pad);
    KPR_GUARD(cin <= 4294967295U / b);
    uint32_t xy = b * cin;
    KPR_GUARD(h_in <= 4294967295U / xy);
    uint32_t wxy = xy * h_in;
    KPR_GUARD(w_in <= 4294967295U / wxy);
    KRML_HOST_IGNORE(wxy * w_in);
    KPR_GUARD(cin <= 4294967295U / cout);
    uint32_t xy0 = cout * cin;
    KPR_GUARD(kh <= 4294967295U / xy0);
    uint32_t wxy0 = xy0 * kh;
    KPR_GUARD(kw <= 4294967295U / wxy0);
    KRML_HOST_IGNORE(wxy0 * kw);
    KPR_GUARD(kh <= 4294967295U / cin);
    uint32_t xy1 = cin * kh;
    KPR_GUARD(kw <= 4294967295U / xy1);
    KRML_HOST_IGNORE(xy1 * kw);
    KPR_GUARD(w_out0 <= 4294967295U / h_out);
    KRML_HOST_IGNORE(h_out * w_out0);
    KPR_GUARD(h_out <= 4294967295U / cout);
    uint32_t xy2 = cout * h_out;
    KPR_GUARD(w_out0 <= 4294967295U / xy2);
    KRML_HOST_IGNORE(xy2 * w_out0);
    KPR_GUARD(cout <= 4294967295U / b);
    uint32_t xy3 = b * cout;
    KPR_GUARD(h_out <= 4294967295U / xy3);
    uint32_t wxy1 = xy3 * h_out;
    KPR_GUARD(w_out0 <= 4294967295U / wxy1);
    uint32_t ylen = wxy1 * w_out0;
    KPR_GUARD(stride <= 4294967295U / h_out);
    uint32_t hs = h_out * stride;
    KPR_GUARD(hs <= 4294967295U - kh);
    KRML_HOST_IGNORE(hs + kh);
    KPR_GUARD(stride <= 4294967295U / w_out0);
    uint32_t ws = w_out0 * stride;
    KPR_GUARD(ws <= 4294967295U - kw);
    KRML_HOST_IGNORE(ws + kw);
    KPR_GUARD(ylen <= 2147483648U);
    uint32_t h_out0 =
        Kuiper_KB_Conv2DAlloc_conv2d_out_dim(h_in, kh, stride, pad);
    uint32_t w_out =
        Kuiper_KB_Conv2DAlloc_conv2d_out_dim(w_in, kw, stride, pad);
    return (KRML_CLITERAL(Kuiper_KB_Conv2DAlloc_conv2d_raw_result){
        .h_out = h_out0,
        .w_out = w_out,
        .output = Kuiper_KB_Conv2DAlloc_conv2d_general_alloc_f32(
            b, cin, h_in, w_in, cout, kh, kw, stride, pad, h_out0, w_out, gx,
            gw, gbias)});
}

__global__
/**
  hoisted when extracting conv2d_raw_alloc_zero_f32
*/
static void
__hoisted_conv2d_raw_alloc_zero_f32_0(uint32_t cout, float *gbias)
{
    if (1024U * blockIdx.x + threadIdx.x < cout)
        gbias[1024U * blockIdx.x + threadIdx.x] = 0.0f;
}

Kuiper_KB_Conv2DAlloc_conv2d_raw_result
Kuiper_KB_Conv2DAlloc_conv2d_raw_alloc_zero_f32(uint32_t b, uint32_t cin,
                                                uint32_t h_in, uint32_t w_in,
                                                uint32_t cout, uint32_t kh,
                                                uint32_t kw, uint32_t stride,
                                                uint32_t pad, float *gx,
                                                float *gw)
{
    KPR_GUARD(pad <= 2147483647U);
    uint32_t two_pad = 2U * pad;
    KPR_GUARD(h_in <= 4294967295U - two_pad);
    uint32_t h_pad = h_in + two_pad;
    KPR_GUARD(w_in <= 4294967295U - two_pad);
    uint32_t w_pad = w_in + two_pad;
    KPR_GUARD(kh <= h_pad);
    KPR_GUARD(kw <= w_pad);
    uint32_t h_out =
        Kuiper_KB_Conv2DAlloc_conv2d_out_dim(h_in, kh, stride, pad);
    uint32_t w_out =
        Kuiper_KB_Conv2DAlloc_conv2d_out_dim(w_in, kw, stride, pad);
    KPR_GUARD(cin <= 4294967295U / b);
    uint32_t xy = b * cin;
    KPR_GUARD(h_in <= 4294967295U / xy);
    uint32_t wxy = xy * h_in;
    KPR_GUARD(w_in <= 4294967295U / wxy);
    KRML_HOST_IGNORE(wxy * w_in);
    KPR_GUARD(cin <= 4294967295U / cout);
    uint32_t xy0 = cout * cin;
    KPR_GUARD(kh <= 4294967295U / xy0);
    uint32_t wxy0 = xy0 * kh;
    KPR_GUARD(kw <= 4294967295U / wxy0);
    KRML_HOST_IGNORE(wxy0 * kw);
    KPR_GUARD(kh <= 4294967295U / cin);
    uint32_t xy1 = cin * kh;
    KPR_GUARD(kw <= 4294967295U / xy1);
    KRML_HOST_IGNORE(xy1 * kw);
    KPR_GUARD(w_out <= 4294967295U / h_out);
    KRML_HOST_IGNORE(h_out * w_out);
    KPR_GUARD(h_out <= 4294967295U / cout);
    uint32_t xy2 = cout * h_out;
    KPR_GUARD(w_out <= 4294967295U / xy2);
    KRML_HOST_IGNORE(xy2 * w_out);
    KPR_GUARD(cout <= 4294967295U / b);
    uint32_t xy3 = b * cout;
    KPR_GUARD(h_out <= 4294967295U / xy3);
    uint32_t wxy1 = xy3 * h_out;
    KPR_GUARD(w_out <= 4294967295U / wxy1);
    uint32_t ylen = wxy1 * w_out;
    KPR_GUARD(stride <= 4294967295U / h_out);
    uint32_t hs = h_out * stride;
    KPR_GUARD(hs <= 4294967295U - kh);
    KRML_HOST_IGNORE(hs + kh);
    KPR_GUARD(stride <= 4294967295U / w_out);
    uint32_t ws = w_out * stride;
    KPR_GUARD(ws <= 4294967295U - kw);
    KRML_HOST_IGNORE(ws + kw);
    KPR_GUARD(ylen <= 2147483648U);
    float *gbias = (float *) KPR_GPU_ALLOC(sizeof(float), cout);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_conv2d_raw_alloc_zero_f32_0,
              cout / 1024U + (uint32_t) (cout % 1024U != 0U), 1024U, 0U, s,
              cout, gbias);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    Kuiper_KB_Conv2DAlloc_conv2d_raw_result r =
        Kuiper_KB_Conv2DAlloc_conv2d_raw_alloc_bias_f32(
            b, cin, h_in, w_in, cout, kh, kw, stride, pad, gx, gw, gbias);
    MUST(cudaFree(gbias));
    return r;
}
