
#include "Kuiper_KB_Conv3DAlloc.h"

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

static void conv3d_general_f32(uint32_t b, uint32_t cin, uint32_t d_in,
                               uint32_t h_in, uint32_t w_in, uint32_t cout,
                               uint32_t kd, uint32_t kh, uint32_t kw,
                               uint32_t stride, uint32_t pad, uint32_t d_out,
                               uint32_t h_out, uint32_t w_out, float *gx,
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

uint32_t Kuiper_KB_Conv3DAlloc_conv3d_out_dim(uint32_t n, uint32_t k,
                                              uint32_t stride, uint32_t pad)
{
    return (n + 2U * pad - k) / stride + 1U;
}

float *Kuiper_KB_Conv3DAlloc_conv3d_general_alloc_f32(
    uint32_t b, uint32_t cin, uint32_t d_in, uint32_t h_in, uint32_t w_in,
    uint32_t cout, uint32_t kd, uint32_t kh, uint32_t kw, uint32_t stride,
    uint32_t pad, uint32_t d_out, uint32_t h_out, uint32_t w_out, float *gx,
    float *gw, float *gbias)
{
    float *gy = (float *) KPR_GPU_ALLOC(sizeof(float),
                                        b * cout * d_out * h_out * w_out);
    conv3d_general_f32(b, cin, d_in, h_in, w_in, cout, kd, kh, kw, stride, pad,
                       d_out, h_out, w_out, gx, gw, gbias, gy);
    return gy;
}

Kuiper_KB_Conv3DAlloc_conv3d_raw_result
Kuiper_KB_Conv3DAlloc_conv3d_raw_alloc_bias_f32(
    uint32_t b, uint32_t cin, uint32_t d_in, uint32_t h_in, uint32_t w_in,
    uint32_t cout, uint32_t kd, uint32_t kh, uint32_t kw, uint32_t stride,
    uint32_t pad, float *gx, float *gw, float *gbias)
{
    KPR_GUARD(pad <= 2147483647U);
    uint32_t two_pad = 2U * pad;
    KPR_GUARD(d_in <= 4294967295U - two_pad);
    uint32_t d_pad = d_in + two_pad;
    KPR_GUARD(h_in <= 4294967295U - two_pad);
    uint32_t h_pad = h_in + two_pad;
    KPR_GUARD(w_in <= 4294967295U - two_pad);
    uint32_t w_pad = w_in + two_pad;
    KPR_GUARD(kd <= d_pad);
    KPR_GUARD(kh <= h_pad);
    KPR_GUARD(kw <= w_pad);
    uint32_t d_out =
        Kuiper_KB_Conv3DAlloc_conv3d_out_dim(d_in, kd, stride, pad);
    uint32_t h_out0 =
        Kuiper_KB_Conv3DAlloc_conv3d_out_dim(h_in, kh, stride, pad);
    uint32_t w_out0 =
        Kuiper_KB_Conv3DAlloc_conv3d_out_dim(w_in, kw, stride, pad);
    KPR_GUARD(cin <= 4294967295U / b);
    uint32_t xy0 = b * cin;
    KPR_GUARD(d_in <= 4294967295U / xy0);
    uint32_t wxy = xy0 * d_in;
    KPR_GUARD(h_in <= 4294967295U / wxy);
    uint32_t vwxy = wxy * h_in;
    KPR_GUARD(w_in <= 4294967295U / vwxy);
    KRML_HOST_IGNORE(vwxy * w_in);
    KPR_GUARD(cin <= 4294967295U / cout);
    uint32_t xy1 = cout * cin;
    KPR_GUARD(kd <= 4294967295U / xy1);
    uint32_t wxy0 = xy1 * kd;
    KPR_GUARD(kh <= 4294967295U / wxy0);
    uint32_t vwxy0 = wxy0 * kh;
    KPR_GUARD(kw <= 4294967295U / vwxy0);
    KRML_HOST_IGNORE(vwxy0 * kw);
    KPR_GUARD(cout <= 4294967295U / b);
    uint32_t xy2 = b * cout;
    KPR_GUARD(d_out <= 4294967295U / xy2);
    uint32_t wxy1 = xy2 * d_out;
    KPR_GUARD(h_out0 <= 4294967295U / wxy1);
    uint32_t vwxy1 = wxy1 * h_out0;
    KPR_GUARD(w_out0 <= 4294967295U / vwxy1);
    uint32_t ylen = vwxy1 * w_out0;
    KPR_GUARD(kd <= 4294967295U / cin);
    uint32_t xy = cin * kd;
    KPR_GUARD(kh <= 4294967295U / xy);
    uint32_t wxy2 = xy * kh;
    KPR_GUARD(kw <= 4294967295U / wxy2);
    KRML_HOST_IGNORE(wxy2 * kw);
    KPR_GUARD(kh <= 4294967295U / kd);
    uint32_t xy3 = kd * kh;
    KPR_GUARD(kw <= 4294967295U / xy3);
    KRML_HOST_IGNORE(xy3 * kw);
    KPR_GUARD(kw <= 4294967295U / kh);
    KRML_HOST_IGNORE(kh * kw);
    KPR_GUARD(w_out0 <= 4294967295U / h_out0);
    KRML_HOST_IGNORE(h_out0 * w_out0);
    KPR_GUARD(h_out0 <= 4294967295U / d_out);
    uint32_t xy4 = d_out * h_out0;
    KPR_GUARD(w_out0 <= 4294967295U / xy4);
    KRML_HOST_IGNORE(xy4 * w_out0);
    KPR_GUARD(d_out <= 4294967295U / cout);
    uint32_t xy5 = cout * d_out;
    KPR_GUARD(h_out0 <= 4294967295U / xy5);
    uint32_t wxy3 = xy5 * h_out0;
    KPR_GUARD(w_out0 <= 4294967295U / wxy3);
    KRML_HOST_IGNORE(wxy3 * w_out0);
    KPR_GUARD(stride <= 4294967295U / d_out);
    uint32_t ds = d_out * stride;
    KPR_GUARD(ds <= 4294967295U - kd);
    KRML_HOST_IGNORE(ds + kd);
    KPR_GUARD(stride <= 4294967295U / h_out0);
    uint32_t hs = h_out0 * stride;
    KPR_GUARD(hs <= 4294967295U - kh);
    KRML_HOST_IGNORE(hs + kh);
    KPR_GUARD(stride <= 4294967295U / w_out0);
    uint32_t ws = w_out0 * stride;
    KPR_GUARD(ws <= 4294967295U - kw);
    KRML_HOST_IGNORE(ws + kw);
    KPR_GUARD(ylen <= 2147483648U);
    uint32_t d_out0 =
        Kuiper_KB_Conv3DAlloc_conv3d_out_dim(d_in, kd, stride, pad);
    uint32_t h_out =
        Kuiper_KB_Conv3DAlloc_conv3d_out_dim(h_in, kh, stride, pad);
    uint32_t w_out =
        Kuiper_KB_Conv3DAlloc_conv3d_out_dim(w_in, kw, stride, pad);
    float *gy = (float *) KPR_GPU_ALLOC(sizeof(float),
                                        b * cout * d_out0 * h_out * w_out);
    conv3d_general_f32(b, cin, d_in, h_in, w_in, cout, kd, kh, kw, stride, pad,
                       d_out0, h_out, w_out, gx, gw, gbias, gy);
    return (KRML_CLITERAL(Kuiper_KB_Conv3DAlloc_conv3d_raw_result){
        .d_out = d_out0, .h_out = h_out, .w_out = w_out, .output = gy});
}

__global__
/**
  hoisted when extracting conv3d_raw_alloc_zero_f32
*/
static void
__hoisted_conv3d_raw_alloc_zero_f32_0(uint32_t cout, float *gbias)
{
    if (1024U * blockIdx.x + threadIdx.x < cout)
        gbias[1024U * blockIdx.x + threadIdx.x] = 0.0f;
}

Kuiper_KB_Conv3DAlloc_conv3d_raw_result
Kuiper_KB_Conv3DAlloc_conv3d_raw_alloc_zero_f32(
    uint32_t b, uint32_t cin, uint32_t d_in, uint32_t h_in, uint32_t w_in,
    uint32_t cout, uint32_t kd, uint32_t kh, uint32_t kw, uint32_t stride,
    uint32_t pad, float *gx, float *gw)
{
    KPR_GUARD(pad <= 2147483647U);
    uint32_t two_pad = 2U * pad;
    KPR_GUARD(d_in <= 4294967295U - two_pad);
    uint32_t d_pad = d_in + two_pad;
    KPR_GUARD(h_in <= 4294967295U - two_pad);
    uint32_t h_pad = h_in + two_pad;
    KPR_GUARD(w_in <= 4294967295U - two_pad);
    uint32_t w_pad = w_in + two_pad;
    KPR_GUARD(kd <= d_pad);
    KPR_GUARD(kh <= h_pad);
    KPR_GUARD(kw <= w_pad);
    uint32_t d_out =
        Kuiper_KB_Conv3DAlloc_conv3d_out_dim(d_in, kd, stride, pad);
    uint32_t h_out0 =
        Kuiper_KB_Conv3DAlloc_conv3d_out_dim(h_in, kh, stride, pad);
    uint32_t w_out0 =
        Kuiper_KB_Conv3DAlloc_conv3d_out_dim(w_in, kw, stride, pad);
    KPR_GUARD(cin <= 4294967295U / b);
    uint32_t xy0 = b * cin;
    KPR_GUARD(d_in <= 4294967295U / xy0);
    uint32_t wxy = xy0 * d_in;
    KPR_GUARD(h_in <= 4294967295U / wxy);
    uint32_t vwxy = wxy * h_in;
    KPR_GUARD(w_in <= 4294967295U / vwxy);
    KRML_HOST_IGNORE(vwxy * w_in);
    KPR_GUARD(cin <= 4294967295U / cout);
    uint32_t xy1 = cout * cin;
    KPR_GUARD(kd <= 4294967295U / xy1);
    uint32_t wxy0 = xy1 * kd;
    KPR_GUARD(kh <= 4294967295U / wxy0);
    uint32_t vwxy0 = wxy0 * kh;
    KPR_GUARD(kw <= 4294967295U / vwxy0);
    KRML_HOST_IGNORE(vwxy0 * kw);
    KPR_GUARD(cout <= 4294967295U / b);
    uint32_t xy2 = b * cout;
    KPR_GUARD(d_out <= 4294967295U / xy2);
    uint32_t wxy1 = xy2 * d_out;
    KPR_GUARD(h_out0 <= 4294967295U / wxy1);
    uint32_t vwxy1 = wxy1 * h_out0;
    KPR_GUARD(w_out0 <= 4294967295U / vwxy1);
    uint32_t ylen = vwxy1 * w_out0;
    KPR_GUARD(kd <= 4294967295U / cin);
    uint32_t xy = cin * kd;
    KPR_GUARD(kh <= 4294967295U / xy);
    uint32_t wxy2 = xy * kh;
    KPR_GUARD(kw <= 4294967295U / wxy2);
    KRML_HOST_IGNORE(wxy2 * kw);
    KPR_GUARD(kh <= 4294967295U / kd);
    uint32_t xy3 = kd * kh;
    KPR_GUARD(kw <= 4294967295U / xy3);
    KRML_HOST_IGNORE(xy3 * kw);
    KPR_GUARD(kw <= 4294967295U / kh);
    KRML_HOST_IGNORE(kh * kw);
    KPR_GUARD(w_out0 <= 4294967295U / h_out0);
    KRML_HOST_IGNORE(h_out0 * w_out0);
    KPR_GUARD(h_out0 <= 4294967295U / d_out);
    uint32_t xy4 = d_out * h_out0;
    KPR_GUARD(w_out0 <= 4294967295U / xy4);
    KRML_HOST_IGNORE(xy4 * w_out0);
    KPR_GUARD(d_out <= 4294967295U / cout);
    uint32_t xy5 = cout * d_out;
    KPR_GUARD(h_out0 <= 4294967295U / xy5);
    uint32_t wxy3 = xy5 * h_out0;
    KPR_GUARD(w_out0 <= 4294967295U / wxy3);
    KRML_HOST_IGNORE(wxy3 * w_out0);
    KPR_GUARD(stride <= 4294967295U / d_out);
    uint32_t ds = d_out * stride;
    KPR_GUARD(ds <= 4294967295U - kd);
    KRML_HOST_IGNORE(ds + kd);
    KPR_GUARD(stride <= 4294967295U / h_out0);
    uint32_t hs = h_out0 * stride;
    KPR_GUARD(hs <= 4294967295U - kh);
    KRML_HOST_IGNORE(hs + kh);
    KPR_GUARD(stride <= 4294967295U / w_out0);
    uint32_t ws = w_out0 * stride;
    KPR_GUARD(ws <= 4294967295U - kw);
    KRML_HOST_IGNORE(ws + kw);
    KPR_GUARD(ylen <= 2147483648U);
    float *gbias = (float *) KPR_GPU_ALLOC(sizeof(float), cout);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_conv3d_raw_alloc_zero_f32_0,
              cout / 1024U + (uint32_t) (cout % 1024U != 0U), 1024U, 0U, s,
              cout, gbias);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    uint32_t d_out0 =
        Kuiper_KB_Conv3DAlloc_conv3d_out_dim(d_in, kd, stride, pad);
    uint32_t h_out =
        Kuiper_KB_Conv3DAlloc_conv3d_out_dim(h_in, kh, stride, pad);
    uint32_t w_out =
        Kuiper_KB_Conv3DAlloc_conv3d_out_dim(w_in, kw, stride, pad);
    float *gy = (float *) KPR_GPU_ALLOC(sizeof(float),
                                        b * cout * d_out0 * h_out * w_out);
    conv3d_general_f32(b, cin, d_in, h_in, w_in, cout, kd, kh, kw, stride, pad,
                       d_out0, h_out, w_out, gx, gw, gbias, gy);
    float *gy0 = gy;
    MUST(cudaFree(gbias));
    return (KRML_CLITERAL(Kuiper_KB_Conv3DAlloc_conv3d_raw_result){
        .d_out = d_out0, .h_out = h_out, .w_out = w_out, .output = gy0});
}
