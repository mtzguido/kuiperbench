
#include "Kuiper_KB_ConvT1DGeneral.h"

static uint32_t convt_out_dim(uint32_t n, uint32_t s, uint32_t d, uint32_t k,
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

static void convt2d_general_f32(uint32_t b, uint32_t cin, uint32_t h_in,
                                uint32_t w_in, uint32_t cout, uint32_t kh,
                                uint32_t kw, uint32_t sh, uint32_t sw,
                                uint32_t ph, uint32_t pw, uint32_t dh,
                                uint32_t dw, uint32_t h_out, uint32_t w_out,
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

static float *convt2d_general_alloc_f32(uint32_t b, uint32_t cin, uint32_t h_in,
                                        uint32_t w_in, uint32_t cout,
                                        uint32_t kh, uint32_t kw, uint32_t sh,
                                        uint32_t sw, uint32_t ph, uint32_t pw,
                                        uint32_t dh, uint32_t dw,
                                        uint32_t h_out, uint32_t w_out,
                                        float *gx, float *gw, float *gbias)
{
    float *gy =
        (float *) KPR_GPU_ALLOC(sizeof(float), b * cout * h_out * w_out);
    convt2d_general_f32(b, cin, h_in, w_in, cout, kh, kw, sh, sw, ph, pw, dh,
                        dw, h_out, w_out, gx, gw, gbias, gy);
    return gy;
}

__global__
/**
  hoisted when extracting convt1d_general_alloc_f32
*/
static void
__hoisted_convt1d_general_alloc_f32_0(uint32_t cout, float *gbias)
{
    if (1024U * blockIdx.x + threadIdx.x < cout)
        gbias[1024U * blockIdx.x + threadIdx.x] = 0.0f;
}

Prims_dtuple2__uint32_t__float_
Kuiper_KB_ConvT1DGeneral_convt1d_general_alloc_f32(
    uint32_t b, uint32_t cin, uint32_t l_in, uint32_t cout, uint32_t k,
    uint32_t s, uint32_t p, uint32_t opad, uint32_t d, float *gx, float *gw)
{
    KPR_GUARD(opad < s || opad < d);
    uint32_t lm1 = l_in - 1U;
    uint32_t km1 = k - 1U;
    KPR_GUARD(lm1 <= 4294967295U / s);
    uint32_t ls = s * lm1;
    KPR_GUARD(km1 <= 4294967295U / d);
    uint32_t dk = d * km1;
    KPR_GUARD(ls <= 4294967295U - dk);
    uint32_t sum0 = ls + dk;
    KPR_GUARD(sum0 <= 4294967295U - opad);
    uint32_t sum1 = sum0 + opad;
    KPR_GUARD(sum1 <= 4294967295U - 1U);
    uint32_t pos = sum1 + 1U;
    KPR_GUARD(p <= 2147483647U);
    KPR_GUARD(2U * p < pos);
    uint32_t l_out = convt_out_dim(l_in, s, d, k, p, opad);
    KPR_GUARD(cin <= 4294967295U / b);
    uint32_t xy = b * cin;
    KPR_GUARD(l_in <= 4294967295U / xy);
    KRML_HOST_IGNORE(xy * l_in);
    KPR_GUARD(cout <= 4294967295U / cin);
    uint32_t xy0 = cin * cout;
    KPR_GUARD(k <= 4294967295U / xy0);
    KRML_HOST_IGNORE(xy0 * k);
    KPR_GUARD(cout <= 4294967295U / b);
    uint32_t xy1 = b * cout;
    KPR_GUARD(l_out <= 4294967295U / xy1);
    uint32_t ylen = xy1 * l_out;
    KPR_GUARD(k <= 4294967295U / cin);
    KRML_HOST_IGNORE(cin * k);
    KPR_GUARD(l_out <= 4294967295U / cout);
    KRML_HOST_IGNORE(cout * l_out);
    KPR_GUARD(l_out <= 4294967295U - p);
    KRML_HOST_IGNORE(l_out + p);
    KPR_GUARD(d <= 4294967295U / k);
    KRML_HOST_IGNORE(k * d);
    KPR_GUARD(ylen <= 2147483648U);
    uint32_t l_out1 = convt_out_dim(l_in, s, d, k, p, opad);
    float *gbias = (float *) KPR_GPU_ALLOC(sizeof(float), cout);
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_convt1d_general_alloc_f32_0,
              cout / 1024U + (uint32_t) (cout % 1024U != 0U), 1024U, 0U, s1,
              cout, gbias);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
    float *gy =
        convt2d_general_alloc_f32(b, cin, 1U, l_in, cout, 1U, k, 1U, s, 0U, p,
                                  1U, d, 1U, l_out1, gx, gw, gbias);
    MUST(cudaFree(gbias));
    return (KRML_CLITERAL(Prims_dtuple2__uint32_t__float_){.fst = l_out1,
                                                           .snd = gy});
}
