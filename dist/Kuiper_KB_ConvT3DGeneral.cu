
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
        for (; k < n_taps; k = kk + 1U) {
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
