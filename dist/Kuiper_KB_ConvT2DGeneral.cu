
#include "Kuiper_KB_ConvT2DGeneral.h"

uint32_t Kuiper_KB_ConvT2DGeneral_convt_out_dim(uint32_t n, uint32_t s,
                                                uint32_t d, uint32_t k,
                                                uint32_t p, uint32_t opad)
{
    return (n - 1U) * s + d * (k - 1U) + opad + 1U - 2U * p;
}

__global__
/**
  hoisted when extracting convt2d_general_f32
*/
static void
__hoisted_convt2d_general_f32_0(uint32_t b, uint32_t cin, uint32_t h_in,
                                uint32_t w_in, uint32_t cout, uint32_t kh,
                                uint32_t kw, uint32_t sh, uint32_t sw,
                                uint32_t ph, uint32_t pw, uint32_t dh,
                                uint32_t dw, uint32_t h_out, uint32_t w_out,
                                float *gx, float *gw, float *gbias, float *gy)
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
        uint32_t oh_ph = r2 / w_out + ph;
        uint32_t ow_pw = r2 % w_out + pw;
        float acc = 0.0f;
        uint32_t k = 0U;
        for (; k < n_taps; k++) {
            uint32_t kk = k;
            uint32_t ic = kk / kh_kw;
            uint32_t r = kk % kh_kw;
            uint32_t kh_i = r / kw;
            uint32_t kw_i = r % kw;
            uint32_t kh_dh = kh_i * dh;
            uint32_t kw_dw = kw_i * dw;
            float ite;
            if (oh_ph >= kh_dh && ow_pw >= kw_dw) {
                uint32_t h_num = oh_ph - kh_dh;
                uint32_t w_num = ow_pw - kw_dw;
                if (h_num % sh == 0U && w_num % sw == 0U) {
                    uint32_t hi = h_num / sh;
                    uint32_t wi = w_num / sw;
                    ite = hi < h_in && wi < w_in
                              ? gx[((bi * cin + ic) * h_in + hi) * w_in + wi]
                              : 0.0f;
                } else
                    ite = 0.0f;
            } else
                ite = 0.0f;
            acc += ite * gw[((ic * cout + oc) * kh + kh_i) * kw + kw_i];
        }
        gy[1024U * blockIdx.x + threadIdx.x] = gbias[oc] + acc;
    }
}

void Kuiper_KB_ConvT2DGeneral_convt2d_general_f32(
    uint32_t b, uint32_t cin, uint32_t h_in, uint32_t w_in, uint32_t cout,
    uint32_t kh, uint32_t kw, uint32_t sh, uint32_t sw, uint32_t ph,
    uint32_t pw, uint32_t dh, uint32_t dw, uint32_t h_out, uint32_t w_out,
    float *gx, float *gw, float *gbias, float *gy)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_convt2d_general_f32_0,
              b * cout * h_out * w_out / 1024U +
                  (uint32_t) (b * cout * h_out * w_out % 1024U != 0U),
              1024U, 0U, s, b, cin, h_in, w_in, cout, kh, kw, sh, sw, ph, pw,
              dh, dw, h_out, w_out, gx, gw, gbias, gy);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

float *Kuiper_KB_ConvT2DGeneral_convt2d_general_alloc_f32(
    uint32_t b, uint32_t cin, uint32_t h_in, uint32_t w_in, uint32_t cout,
    uint32_t kh, uint32_t kw, uint32_t sh, uint32_t sw, uint32_t ph,
    uint32_t pw, uint32_t dh, uint32_t dw, uint32_t h_out, uint32_t w_out,
    float *gx, float *gw, float *gbias)
{
    float *gy =
        (float *) KPR_GPU_ALLOC(sizeof(float), b * cout * h_out * w_out);
    Kuiper_KB_ConvT2DGeneral_convt2d_general_f32(
        b, cin, h_in, w_in, cout, kh, kw, sh, sw, ph, pw, dh, dw, h_out, w_out,
        gx, gw, gbias, gy);
    return gy;
}

Kuiper_KB_ConvT2DGeneral_convt2d_raw_result
Kuiper_KB_ConvT2DGeneral_convt2d_raw_alloc_bias_f32(
    uint32_t b, uint32_t cin, uint32_t h_in, uint32_t w_in, uint32_t cout,
    uint32_t kh, uint32_t kw, uint32_t sh, uint32_t sw, uint32_t ph,
    uint32_t pw, uint32_t oph, uint32_t opw, uint32_t dh, uint32_t dw,
    float *gx, float *gw, float *gbias)
{
    KPR_GUARD(oph < sh || oph < dh);
    KPR_GUARD(opw < sw || opw < dw);
    uint32_t hm1 = h_in - 1U;
    uint32_t khm1 = kh - 1U;
    KPR_GUARD(hm1 <= 4294967295U / sh);
    uint32_t hs = sh * hm1;
    KPR_GUARD(khm1 <= 4294967295U / dh);
    uint32_t hdk = dh * khm1;
    KPR_GUARD(hs <= 4294967295U - hdk);
    uint32_t hsum0 = hs + hdk;
    KPR_GUARD(hsum0 <= 4294967295U - oph);
    uint32_t hsum1 = hsum0 + oph;
    KPR_GUARD(hsum1 <= 4294967295U - 1U);
    uint32_t hpos = hsum1 + 1U;
    KPR_GUARD(ph <= 2147483647U);
    KPR_GUARD(2U * ph < hpos);
    uint32_t wm1 = w_in - 1U;
    uint32_t kwm1 = kw - 1U;
    KPR_GUARD(wm1 <= 4294967295U / sw);
    uint32_t ws = sw * wm1;
    KPR_GUARD(kwm1 <= 4294967295U / dw);
    uint32_t wdk = dw * kwm1;
    KPR_GUARD(ws <= 4294967295U - wdk);
    uint32_t wsum0 = ws + wdk;
    KPR_GUARD(wsum0 <= 4294967295U - opw);
    uint32_t wsum1 = wsum0 + opw;
    KPR_GUARD(wsum1 <= 4294967295U - 1U);
    uint32_t wpos = wsum1 + 1U;
    KPR_GUARD(pw <= 2147483647U);
    KPR_GUARD(2U * pw < wpos);
    uint32_t h_out =
        Kuiper_KB_ConvT2DGeneral_convt_out_dim(h_in, sh, dh, kh, ph, oph);
    uint32_t w_out0 =
        Kuiper_KB_ConvT2DGeneral_convt_out_dim(w_in, sw, dw, kw, pw, opw);
    KPR_GUARD(cin <= 4294967295U / b);
    uint32_t xy0 = b * cin;
    KPR_GUARD(h_in <= 4294967295U / xy0);
    uint32_t wxy = xy0 * h_in;
    KPR_GUARD(w_in <= 4294967295U / wxy);
    KRML_HOST_IGNORE(wxy * w_in);
    KPR_GUARD(cout <= 4294967295U / cin);
    uint32_t xy1 = cin * cout;
    KPR_GUARD(kh <= 4294967295U / xy1);
    uint32_t wxy0 = xy1 * kh;
    KPR_GUARD(kw <= 4294967295U / wxy0);
    KRML_HOST_IGNORE(wxy0 * kw);
    KPR_GUARD(cout <= 4294967295U / b);
    uint32_t xy2 = b * cout;
    KPR_GUARD(h_out <= 4294967295U / xy2);
    uint32_t wxy1 = xy2 * h_out;
    KPR_GUARD(w_out0 <= 4294967295U / wxy1);
    uint32_t ylen = wxy1 * w_out0;
    KPR_GUARD(kh <= 4294967295U / cin);
    uint32_t xy = cin * kh;
    KPR_GUARD(kw <= 4294967295U / xy);
    KRML_HOST_IGNORE(xy * kw);
    KPR_GUARD(kw <= 4294967295U / kh);
    KRML_HOST_IGNORE(kh * kw);
    KPR_GUARD(w_out0 <= 4294967295U / h_out);
    KRML_HOST_IGNORE(h_out * w_out0);
    KPR_GUARD(h_out <= 4294967295U / cout);
    uint32_t xy3 = cout * h_out;
    KPR_GUARD(w_out0 <= 4294967295U / xy3);
    KRML_HOST_IGNORE(xy3 * w_out0);
    KPR_GUARD(h_out <= 4294967295U - ph);
    KRML_HOST_IGNORE(h_out + ph);
    KPR_GUARD(w_out0 <= 4294967295U - pw);
    KRML_HOST_IGNORE(w_out0 + pw);
    KPR_GUARD(dh <= 4294967295U / kh);
    KRML_HOST_IGNORE(kh * dh);
    KPR_GUARD(dw <= 4294967295U / kw);
    KRML_HOST_IGNORE(kw * dw);
    KPR_GUARD(ylen <= 2147483648U);
    uint32_t h_out0 =
        Kuiper_KB_ConvT2DGeneral_convt_out_dim(h_in, sh, dh, kh, ph, oph);
    uint32_t w_out =
        Kuiper_KB_ConvT2DGeneral_convt_out_dim(w_in, sw, dw, kw, pw, opw);
    return (KRML_CLITERAL(Kuiper_KB_ConvT2DGeneral_convt2d_raw_result){
        .h_out = h_out0,
        .w_out = w_out,
        .output = Kuiper_KB_ConvT2DGeneral_convt2d_general_alloc_f32(
            b, cin, h_in, w_in, cout, kh, kw, sh, sw, ph, pw, dh, dw, h_out0,
            w_out, gx, gw, gbias)});
}

__global__
/**
  hoisted when extracting convt2d_raw_alloc_zero_f32
*/
static void
__hoisted_convt2d_raw_alloc_zero_f32_0(uint32_t cout, float *gbias)
{
    if (1024U * blockIdx.x + threadIdx.x < cout)
        gbias[1024U * blockIdx.x + threadIdx.x] = 0.0f;
}

Kuiper_KB_ConvT2DGeneral_convt2d_raw_result
Kuiper_KB_ConvT2DGeneral_convt2d_raw_alloc_zero_f32(
    uint32_t b, uint32_t cin, uint32_t h_in, uint32_t w_in, uint32_t cout,
    uint32_t kh, uint32_t kw, uint32_t sh, uint32_t sw, uint32_t ph,
    uint32_t pw, uint32_t oph, uint32_t opw, uint32_t dh, uint32_t dw,
    float *gx, float *gw)
{
    KPR_GUARD(oph < sh || oph < dh);
    KPR_GUARD(opw < sw || opw < dw);
    uint32_t hm1 = h_in - 1U;
    uint32_t khm1 = kh - 1U;
    KPR_GUARD(hm1 <= 4294967295U / sh);
    uint32_t hs = sh * hm1;
    KPR_GUARD(khm1 <= 4294967295U / dh);
    uint32_t hdk = dh * khm1;
    KPR_GUARD(hs <= 4294967295U - hdk);
    uint32_t hsum0 = hs + hdk;
    KPR_GUARD(hsum0 <= 4294967295U - oph);
    uint32_t hsum1 = hsum0 + oph;
    KPR_GUARD(hsum1 <= 4294967295U - 1U);
    uint32_t hpos = hsum1 + 1U;
    KPR_GUARD(ph <= 2147483647U);
    KPR_GUARD(2U * ph < hpos);
    uint32_t wm1 = w_in - 1U;
    uint32_t kwm1 = kw - 1U;
    KPR_GUARD(wm1 <= 4294967295U / sw);
    uint32_t ws = sw * wm1;
    KPR_GUARD(kwm1 <= 4294967295U / dw);
    uint32_t wdk = dw * kwm1;
    KPR_GUARD(ws <= 4294967295U - wdk);
    uint32_t wsum0 = ws + wdk;
    KPR_GUARD(wsum0 <= 4294967295U - opw);
    uint32_t wsum1 = wsum0 + opw;
    KPR_GUARD(wsum1 <= 4294967295U - 1U);
    uint32_t wpos = wsum1 + 1U;
    KPR_GUARD(pw <= 2147483647U);
    KPR_GUARD(2U * pw < wpos);
    uint32_t h_out =
        Kuiper_KB_ConvT2DGeneral_convt_out_dim(h_in, sh, dh, kh, ph, oph);
    uint32_t w_out =
        Kuiper_KB_ConvT2DGeneral_convt_out_dim(w_in, sw, dw, kw, pw, opw);
    KPR_GUARD(cin <= 4294967295U / b);
    uint32_t xy0 = b * cin;
    KPR_GUARD(h_in <= 4294967295U / xy0);
    uint32_t wxy = xy0 * h_in;
    KPR_GUARD(w_in <= 4294967295U / wxy);
    KRML_HOST_IGNORE(wxy * w_in);
    KPR_GUARD(cout <= 4294967295U / cin);
    uint32_t xy1 = cin * cout;
    KPR_GUARD(kh <= 4294967295U / xy1);
    uint32_t wxy0 = xy1 * kh;
    KPR_GUARD(kw <= 4294967295U / wxy0);
    KRML_HOST_IGNORE(wxy0 * kw);
    KPR_GUARD(cout <= 4294967295U / b);
    uint32_t xy2 = b * cout;
    KPR_GUARD(h_out <= 4294967295U / xy2);
    uint32_t wxy1 = xy2 * h_out;
    KPR_GUARD(w_out <= 4294967295U / wxy1);
    uint32_t ylen = wxy1 * w_out;
    KPR_GUARD(kh <= 4294967295U / cin);
    uint32_t xy = cin * kh;
    KPR_GUARD(kw <= 4294967295U / xy);
    KRML_HOST_IGNORE(xy * kw);
    KPR_GUARD(kw <= 4294967295U / kh);
    KRML_HOST_IGNORE(kh * kw);
    KPR_GUARD(w_out <= 4294967295U / h_out);
    KRML_HOST_IGNORE(h_out * w_out);
    KPR_GUARD(h_out <= 4294967295U / cout);
    uint32_t xy3 = cout * h_out;
    KPR_GUARD(w_out <= 4294967295U / xy3);
    KRML_HOST_IGNORE(xy3 * w_out);
    KPR_GUARD(h_out <= 4294967295U - ph);
    KRML_HOST_IGNORE(h_out + ph);
    KPR_GUARD(w_out <= 4294967295U - pw);
    KRML_HOST_IGNORE(w_out + pw);
    KPR_GUARD(dh <= 4294967295U / kh);
    KRML_HOST_IGNORE(kh * dh);
    KPR_GUARD(dw <= 4294967295U / kw);
    KRML_HOST_IGNORE(kw * dw);
    KPR_GUARD(ylen <= 2147483648U);
    float *gbias = (float *) KPR_GPU_ALLOC(sizeof(float), cout);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_convt2d_raw_alloc_zero_f32_0,
              cout / 1024U + (uint32_t) (cout % 1024U != 0U), 1024U, 0U, s,
              cout, gbias);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    Kuiper_KB_ConvT2DGeneral_convt2d_raw_result r =
        Kuiper_KB_ConvT2DGeneral_convt2d_raw_alloc_bias_f32(
            b, cin, h_in, w_in, cout, kh, kw, sh, sw, ph, pw, oph, opw, dh, dw,
            gx, gw, gbias);
    MUST(cudaFree(gbias));
    return r;
}
