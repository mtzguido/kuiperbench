
#include "Kuiper_KB_ConvT3DGeneral.h"

uint32_t Kuiper_KB_ConvT3DGeneral_convt_out_dim(uint32_t n, uint32_t s,
                                                uint32_t d, uint32_t k,
                                                uint32_t p, uint32_t opad)
{
    return (n - 1U) * s + d * (k - 1U) + opad + 1U - 2U * p;
}

__global__
/**
  hoisted when extracting convt3d_general_f32
*/
static void
__hoisted_convt3d_general_f32_0(uint32_t b, uint32_t cin, uint32_t d_in,
                                uint32_t h_in, uint32_t w_in, uint32_t cout,
                                uint32_t kd, uint32_t kh, uint32_t kw,
                                uint32_t sd, uint32_t sh, uint32_t sw,
                                uint32_t pd, uint32_t ph, uint32_t pw,
                                uint32_t dd, uint32_t dh, uint32_t dw,
                                uint32_t d_out, uint32_t h_out, uint32_t w_out,
                                float *gx, float *gw, float *gbias, float *gy)
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
        uint32_t od_pd = r2 / how + pd;
        uint32_t oh_ph = r3 / w_out + ph;
        uint32_t ow_pw = r3 % w_out + pw;
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
            uint32_t kd_dd = kd_i * dd;
            uint32_t kh_dh = kh_i * dh;
            uint32_t kw_dw = kw_i * dw;
            float ite;
            if (od_pd >= kd_dd && oh_ph >= kh_dh && ow_pw >= kw_dw) {
                uint32_t d_num = od_pd - kd_dd;
                uint32_t h_num = oh_ph - kh_dh;
                uint32_t w_num = ow_pw - kw_dw;
                if (d_num % sd == 0U && h_num % sh == 0U && w_num % sw == 0U) {
                    uint32_t di = d_num / sd;
                    uint32_t hi = h_num / sh;
                    uint32_t wi = w_num / sw;
                    ite = di < d_in && hi < h_in && wi < w_in
                              ? gx[(((bi * cin + ic) * d_in + di) * h_in + hi) *
                                       w_in +
                                   wi]
                              : 0.0f;
                } else
                    ite = 0.0f;
            } else
                ite = 0.0f;
            acc += ite *
                   gw[(((ic * cout + oc) * kd + kd_i) * kh + kh_i) * kw + kw_i];
        }
        gy[1024U * blockIdx.x + threadIdx.x] = gbias[oc] + acc;
    }
}

void Kuiper_KB_ConvT3DGeneral_convt3d_general_f32(
    uint32_t b, uint32_t cin, uint32_t d_in, uint32_t h_in, uint32_t w_in,
    uint32_t cout, uint32_t kd, uint32_t kh, uint32_t kw, uint32_t sd,
    uint32_t sh, uint32_t sw, uint32_t pd, uint32_t ph, uint32_t pw,
    uint32_t dd, uint32_t dh, uint32_t dw, uint32_t d_out, uint32_t h_out,
    uint32_t w_out, float *gx, float *gw, float *gbias, float *gy)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_convt3d_general_f32_0,
              b * cout * d_out * h_out * w_out / 1024U +
                  (uint32_t) (b * cout * d_out * h_out * w_out % 1024U != 0U),
              1024U, 0U, s, b, cin, d_in, h_in, w_in, cout, kd, kh, kw, sd, sh,
              sw, pd, ph, pw, dd, dh, dw, d_out, h_out, w_out, gx, gw, gbias,
              gy);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

float *Kuiper_KB_ConvT3DGeneral_convt3d_general_alloc_f32(
    uint32_t b, uint32_t cin, uint32_t d_in, uint32_t h_in, uint32_t w_in,
    uint32_t cout, uint32_t kd, uint32_t kh, uint32_t kw, uint32_t sd,
    uint32_t sh, uint32_t sw, uint32_t pd, uint32_t ph, uint32_t pw,
    uint32_t dd, uint32_t dh, uint32_t dw, uint32_t d_out, uint32_t h_out,
    uint32_t w_out, float *gx, float *gw, float *gbias)
{
    float *gy = (float *) KPR_GPU_ALLOC(sizeof(float),
                                        b * cout * d_out * h_out * w_out);
    Kuiper_KB_ConvT3DGeneral_convt3d_general_f32(
        b, cin, d_in, h_in, w_in, cout, kd, kh, kw, sd, sh, sw, pd, ph, pw, dd,
        dh, dw, d_out, h_out, w_out, gx, gw, gbias, gy);
    return gy;
}

Kuiper_KB_ConvT3DGeneral_convt3d_raw_result
Kuiper_KB_ConvT3DGeneral_convt3d_raw_alloc_bias_f32(
    uint32_t b, uint32_t cin, uint32_t d_in, uint32_t h_in, uint32_t w_in,
    uint32_t cout, uint32_t kd, uint32_t kh, uint32_t kw, uint32_t sd,
    uint32_t sh, uint32_t sw, uint32_t pd, uint32_t ph, uint32_t pw,
    uint32_t opd, uint32_t oph, uint32_t opw, uint32_t dd, uint32_t dh,
    uint32_t dw, float *gx, float *gw, float *gbias)
{
    KPR_GUARD(opd < sd || opd < dd);
    KPR_GUARD(oph < sh || oph < dh);
    KPR_GUARD(opw < sw || opw < dw);
    uint32_t dm1 = d_in - 1U;
    uint32_t kdm1 = kd - 1U;
    KPR_GUARD(dm1 <= 4294967295U / sd);
    uint32_t ds0 = sd * dm1;
    KPR_GUARD(kdm1 <= 4294967295U / dd);
    uint32_t ddk = dd * kdm1;
    KPR_GUARD(ds0 <= 4294967295U - ddk);
    uint32_t dsum0 = ds0 + ddk;
    KPR_GUARD(dsum0 <= 4294967295U - opd);
    uint32_t dsum1 = dsum0 + opd;
    KPR_GUARD(dsum1 <= 4294967295U - 1U);
    uint32_t dpos = dsum1 + 1U;
    KPR_GUARD(pd <= 2147483647U);
    KPR_GUARD(2U * pd < dpos);
    uint32_t hm1 = h_in - 1U;
    uint32_t khm1 = kh - 1U;
    KPR_GUARD(hm1 <= 4294967295U / sh);
    uint32_t hs0 = sh * hm1;
    KPR_GUARD(khm1 <= 4294967295U / dh);
    uint32_t hdk = dh * khm1;
    KPR_GUARD(hs0 <= 4294967295U - hdk);
    uint32_t hsum0 = hs0 + hdk;
    KPR_GUARD(hsum0 <= 4294967295U - oph);
    uint32_t hsum1 = hsum0 + oph;
    KPR_GUARD(hsum1 <= 4294967295U - 1U);
    uint32_t hpos = hsum1 + 1U;
    KPR_GUARD(ph <= 2147483647U);
    KPR_GUARD(2U * ph < hpos);
    uint32_t wm1 = w_in - 1U;
    uint32_t kwm1 = kw - 1U;
    KPR_GUARD(wm1 <= 4294967295U / sw);
    uint32_t ws0 = sw * wm1;
    KPR_GUARD(kwm1 <= 4294967295U / dw);
    uint32_t wdk = dw * kwm1;
    KPR_GUARD(ws0 <= 4294967295U - wdk);
    uint32_t wsum0 = ws0 + wdk;
    KPR_GUARD(wsum0 <= 4294967295U - opw);
    uint32_t wsum1 = wsum0 + opw;
    KPR_GUARD(wsum1 <= 4294967295U - 1U);
    uint32_t wpos = wsum1 + 1U;
    KPR_GUARD(pw <= 2147483647U);
    KPR_GUARD(2U * pw < wpos);
    uint32_t d_out =
        Kuiper_KB_ConvT3DGeneral_convt_out_dim(d_in, sd, dd, kd, pd, opd);
    uint32_t h_out0 =
        Kuiper_KB_ConvT3DGeneral_convt_out_dim(h_in, sh, dh, kh, ph, oph);
    uint32_t w_out0 =
        Kuiper_KB_ConvT3DGeneral_convt_out_dim(w_in, sw, dw, kw, pw, opw);
    KPR_GUARD(cin <= 4294967295U / b);
    uint32_t xy0 = b * cin;
    KPR_GUARD(d_in <= 4294967295U / xy0);
    uint32_t wxy = xy0 * d_in;
    KPR_GUARD(h_in <= 4294967295U / wxy);
    uint32_t vwxy = wxy * h_in;
    KPR_GUARD(w_in <= 4294967295U / vwxy);
    KRML_HOST_IGNORE(vwxy * w_in);
    KPR_GUARD(cout <= 4294967295U / cin);
    uint32_t xy1 = cin * cout;
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
    KPR_GUARD(d_out <= 4294967295U - pd);
    KRML_HOST_IGNORE(d_out + pd);
    KPR_GUARD(h_out0 <= 4294967295U - ph);
    KRML_HOST_IGNORE(h_out0 + ph);
    KPR_GUARD(w_out0 <= 4294967295U - pw);
    KRML_HOST_IGNORE(w_out0 + pw);
    KPR_GUARD(dd <= 4294967295U / kd);
    KRML_HOST_IGNORE(kd * dd);
    KPR_GUARD(dh <= 4294967295U / kh);
    KRML_HOST_IGNORE(kh * dh);
    KPR_GUARD(dw <= 4294967295U / kw);
    KRML_HOST_IGNORE(kw * dw);
    KPR_GUARD(ylen <= 2147483648U);
    uint32_t d_out0 =
        Kuiper_KB_ConvT3DGeneral_convt_out_dim(d_in, sd, dd, kd, pd, opd);
    uint32_t h_out =
        Kuiper_KB_ConvT3DGeneral_convt_out_dim(h_in, sh, dh, kh, ph, oph);
    uint32_t w_out =
        Kuiper_KB_ConvT3DGeneral_convt_out_dim(w_in, sw, dw, kw, pw, opw);
    float *gy = (float *) KPR_GPU_ALLOC(sizeof(float),
                                        b * cout * d_out0 * h_out * w_out);
    Kuiper_KB_ConvT3DGeneral_convt3d_general_f32(
        b, cin, d_in, h_in, w_in, cout, kd, kh, kw, sd, sh, sw, pd, ph, pw, dd,
        dh, dw, d_out0, h_out, w_out, gx, gw, gbias, gy);
    return (KRML_CLITERAL(Kuiper_KB_ConvT3DGeneral_convt3d_raw_result){
        .fst = d_out0,
        .snd = {.fst = h_out, .snd = {.fst = w_out, .snd = gy}}});
}

__global__
/**
  hoisted when extracting convt3d_raw_alloc_zero_f32
*/
static void
__hoisted_convt3d_raw_alloc_zero_f32_0(uint32_t cout, float *gbias)
{
    if (1024U * blockIdx.x + threadIdx.x < cout)
        gbias[1024U * blockIdx.x + threadIdx.x] = 0.0f;
}

Kuiper_KB_ConvT3DGeneral_convt3d_raw_result
Kuiper_KB_ConvT3DGeneral_convt3d_raw_alloc_zero_f32(
    uint32_t b, uint32_t cin, uint32_t d_in, uint32_t h_in, uint32_t w_in,
    uint32_t cout, uint32_t kd, uint32_t kh, uint32_t kw, uint32_t sd,
    uint32_t sh, uint32_t sw, uint32_t pd, uint32_t ph, uint32_t pw,
    uint32_t opd, uint32_t oph, uint32_t opw, uint32_t dd, uint32_t dh,
    uint32_t dw, float *gx, float *gw)
{
    KPR_GUARD(opd < sd || opd < dd);
    KPR_GUARD(oph < sh || oph < dh);
    KPR_GUARD(opw < sw || opw < dw);
    uint32_t dm1 = d_in - 1U;
    uint32_t kdm1 = kd - 1U;
    KPR_GUARD(dm1 <= 4294967295U / sd);
    uint32_t ds0 = sd * dm1;
    KPR_GUARD(kdm1 <= 4294967295U / dd);
    uint32_t ddk = dd * kdm1;
    KPR_GUARD(ds0 <= 4294967295U - ddk);
    uint32_t dsum0 = ds0 + ddk;
    KPR_GUARD(dsum0 <= 4294967295U - opd);
    uint32_t dsum1 = dsum0 + opd;
    KPR_GUARD(dsum1 <= 4294967295U - 1U);
    uint32_t dpos = dsum1 + 1U;
    KPR_GUARD(pd <= 2147483647U);
    KPR_GUARD(2U * pd < dpos);
    uint32_t hm1 = h_in - 1U;
    uint32_t khm1 = kh - 1U;
    KPR_GUARD(hm1 <= 4294967295U / sh);
    uint32_t hs0 = sh * hm1;
    KPR_GUARD(khm1 <= 4294967295U / dh);
    uint32_t hdk = dh * khm1;
    KPR_GUARD(hs0 <= 4294967295U - hdk);
    uint32_t hsum0 = hs0 + hdk;
    KPR_GUARD(hsum0 <= 4294967295U - oph);
    uint32_t hsum1 = hsum0 + oph;
    KPR_GUARD(hsum1 <= 4294967295U - 1U);
    uint32_t hpos = hsum1 + 1U;
    KPR_GUARD(ph <= 2147483647U);
    KPR_GUARD(2U * ph < hpos);
    uint32_t wm1 = w_in - 1U;
    uint32_t kwm1 = kw - 1U;
    KPR_GUARD(wm1 <= 4294967295U / sw);
    uint32_t ws0 = sw * wm1;
    KPR_GUARD(kwm1 <= 4294967295U / dw);
    uint32_t wdk = dw * kwm1;
    KPR_GUARD(ws0 <= 4294967295U - wdk);
    uint32_t wsum0 = ws0 + wdk;
    KPR_GUARD(wsum0 <= 4294967295U - opw);
    uint32_t wsum1 = wsum0 + opw;
    KPR_GUARD(wsum1 <= 4294967295U - 1U);
    uint32_t wpos = wsum1 + 1U;
    KPR_GUARD(pw <= 2147483647U);
    KPR_GUARD(2U * pw < wpos);
    uint32_t d_out =
        Kuiper_KB_ConvT3DGeneral_convt_out_dim(d_in, sd, dd, kd, pd, opd);
    uint32_t h_out0 =
        Kuiper_KB_ConvT3DGeneral_convt_out_dim(h_in, sh, dh, kh, ph, oph);
    uint32_t w_out0 =
        Kuiper_KB_ConvT3DGeneral_convt_out_dim(w_in, sw, dw, kw, pw, opw);
    KPR_GUARD(cin <= 4294967295U / b);
    uint32_t xy0 = b * cin;
    KPR_GUARD(d_in <= 4294967295U / xy0);
    uint32_t wxy = xy0 * d_in;
    KPR_GUARD(h_in <= 4294967295U / wxy);
    uint32_t vwxy = wxy * h_in;
    KPR_GUARD(w_in <= 4294967295U / vwxy);
    KRML_HOST_IGNORE(vwxy * w_in);
    KPR_GUARD(cout <= 4294967295U / cin);
    uint32_t xy1 = cin * cout;
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
    KPR_GUARD(d_out <= 4294967295U - pd);
    KRML_HOST_IGNORE(d_out + pd);
    KPR_GUARD(h_out0 <= 4294967295U - ph);
    KRML_HOST_IGNORE(h_out0 + ph);
    KPR_GUARD(w_out0 <= 4294967295U - pw);
    KRML_HOST_IGNORE(w_out0 + pw);
    KPR_GUARD(dd <= 4294967295U / kd);
    KRML_HOST_IGNORE(kd * dd);
    KPR_GUARD(dh <= 4294967295U / kh);
    KRML_HOST_IGNORE(kh * dh);
    KPR_GUARD(dw <= 4294967295U / kw);
    KRML_HOST_IGNORE(kw * dw);
    KPR_GUARD(ylen <= 2147483648U);
    float *gbias = (float *) KPR_GPU_ALLOC(sizeof(float), cout);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_convt3d_raw_alloc_zero_f32_0,
              cout / 1024U + (uint32_t) (cout % 1024U != 0U), 1024U, 0U, s,
              cout, gbias);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    uint32_t d_out0 =
        Kuiper_KB_ConvT3DGeneral_convt_out_dim(d_in, sd, dd, kd, pd, opd);
    uint32_t h_out =
        Kuiper_KB_ConvT3DGeneral_convt_out_dim(h_in, sh, dh, kh, ph, oph);
    uint32_t w_out =
        Kuiper_KB_ConvT3DGeneral_convt_out_dim(w_in, sw, dw, kw, pw, opw);
    float *gy = (float *) KPR_GPU_ALLOC(sizeof(float),
                                        b * cout * d_out0 * h_out * w_out);
    Kuiper_KB_ConvT3DGeneral_convt3d_general_f32(
        b, cin, d_in, h_in, w_in, cout, kd, kh, kw, sd, sh, sw, pd, ph, pw, dd,
        dh, dw, d_out0, h_out, w_out, gx, gw, gbias, gy);
    float *gy0 = gy;
    MUST(cudaFree(gbias));
    return (KRML_CLITERAL(Kuiper_KB_ConvT3DGeneral_convt3d_raw_result){
        .fst = d_out0,
        .snd = {.fst = h_out, .snd = {.fst = w_out, .snd = gy0}}});
}
