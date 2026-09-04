
#include "Kuiper_KB_Conv1DAlloc.h"

__global__
/**
  hoisted when extracting conv1d_general_f32
*/
static void
__hoisted_conv1d_general_f32_0(uint32_t b, uint32_t cin, uint32_t l_in,
                               uint32_t cout, uint32_t kk, uint32_t stride,
                               uint32_t pad, uint32_t dilation, uint32_t l_out,
                               float *gx, float *gw, float *gbias, float *gy)
{
    if (1024U * blockIdx.x + threadIdx.x < b * cout * l_out) {
        uint32_t cl = cout * l_out;
        uint32_t bi = (1024U * blockIdx.x + threadIdx.x) / cl;
        uint32_t r1 = (1024U * blockIdx.x + threadIdx.x) % cl;
        uint32_t oc = r1 / l_out;
        uint32_t n_taps = cin * kk;
        uint32_t ol_s = r1 % l_out * stride;
        float acc = 0.0f;
        uint32_t k = 0U;
        for (; k < n_taps; k++) {
            uint32_t kk_v = k;
            uint32_t ic = kk_v / kk;
            uint32_t k_i = kk_v % kk;
            uint32_t l_signed = ol_s + k_i * dilation;
            float ite;
            if (pad <= l_signed) {
                uint32_t li = l_signed - pad;
                ite = li < l_in ? gx[(bi * cin + ic) * l_in + li] : 0.0f;
            } else
                ite = 0.0f;
            acc += ite * gw[(oc * cin + ic) * kk + k_i];
        }
        gy[1024U * blockIdx.x + threadIdx.x] = gbias[oc] + acc;
    }
}

static void conv1d_general_f32(uint32_t b, uint32_t cin, uint32_t l_in,
                               uint32_t cout, uint32_t kk, uint32_t stride,
                               uint32_t pad, uint32_t dilation, uint32_t l_out,
                               float *gx, float *gw, float *gbias, float *gy)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_conv1d_general_f32_0,
              b * cout * l_out / 1024U +
                  (uint32_t) (b * cout * l_out % 1024U != 0U),
              1024U, 0U, s, b, cin, l_in, cout, kk, stride, pad, dilation,
              l_out, gx, gw, gbias, gy);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

uint32_t Kuiper_KB_Conv1DAlloc_conv1d_out_dim(uint32_t n, uint32_t k,
                                              uint32_t stride,
                                              uint32_t dilation, uint32_t pad)
{
    return (n + 2U * pad - ((k - 1U) * dilation + 1U)) / stride + 1U;
}

float *Kuiper_KB_Conv1DAlloc_conv1d_general_alloc_f32(
    uint32_t b, uint32_t cin, uint32_t l_in, uint32_t cout, uint32_t kk,
    uint32_t stride, uint32_t pad, uint32_t dilation, uint32_t l_out, float *gx,
    float *gw, float *gbias)
{
    float *gy = (float *) KPR_GPU_ALLOC(sizeof(float), b * cout * l_out);
    conv1d_general_f32(b, cin, l_in, cout, kk, stride, pad, dilation, l_out, gx,
                       gw, gbias, gy);
    return gy;
}

Kuiper_KB_Conv1DAlloc_conv1d_raw_result
Kuiper_KB_Conv1DAlloc_conv1d_raw_alloc_bias_f32(uint32_t b, uint32_t cin,
                                                uint32_t l_in, uint32_t cout,
                                                uint32_t kk, uint32_t stride,
                                                uint32_t pad, uint32_t dilation,
                                                float *gx, float *gw,
                                                float *gbias)
{
    KPR_GUARD(pad <= 2147483647U);
    uint32_t two_pad = 2U * pad;
    KPR_GUARD(l_in <= 4294967295U - two_pad);
    uint32_t padded = l_in + two_pad;
    uint32_t km1 = kk - 1U;
    KPR_GUARD(km1 <= 4294967295U / dilation);
    uint32_t dilated = dilation * km1;
    KPR_GUARD(dilated <= 4294967295U - 1U);
    KPR_GUARD(dilated + 1U <= padded);
    uint32_t l_out =
        Kuiper_KB_Conv1DAlloc_conv1d_out_dim(l_in, kk, stride, dilation, pad);
    KPR_GUARD(cin <= 4294967295U / b);
    uint32_t xy = b * cin;
    KPR_GUARD(l_in <= 4294967295U / xy);
    KRML_HOST_IGNORE(xy * l_in);
    KPR_GUARD(cin <= 4294967295U / cout);
    uint32_t xy0 = cout * cin;
    KPR_GUARD(kk <= 4294967295U / xy0);
    KRML_HOST_IGNORE(xy0 * kk);
    KPR_GUARD(cout <= 4294967295U / b);
    uint32_t xy1 = b * cout;
    KPR_GUARD(l_out <= 4294967295U / xy1);
    uint32_t ylen = xy1 * l_out;
    KPR_GUARD(kk <= 4294967295U / cin);
    KRML_HOST_IGNORE(cin * kk);
    KPR_GUARD(l_out <= 4294967295U / cout);
    KRML_HOST_IGNORE(cout * l_out);
    KPR_GUARD(stride <= 4294967295U / l_out);
    uint32_t ls = l_out * stride;
    KPR_GUARD(dilation <= 4294967295U / kk);
    uint32_t kd = kk * dilation;
    KPR_GUARD(ls <= 4294967295U - kd);
    KRML_HOST_IGNORE(ls + kd);
    KPR_GUARD(ylen <= 2147483648U);
    uint32_t l_out0 =
        Kuiper_KB_Conv1DAlloc_conv1d_out_dim(l_in, kk, stride, dilation, pad);
    return (KRML_CLITERAL(Kuiper_KB_Conv1DAlloc_conv1d_raw_result){
        .l_out = l_out0,
        .output = Kuiper_KB_Conv1DAlloc_conv1d_general_alloc_f32(
            b, cin, l_in, cout, kk, stride, pad, dilation, l_out0, gx, gw,
            gbias)});
}

__global__
/**
  hoisted when extracting conv1d_raw_alloc_zero_f32
*/
static void
__hoisted_conv1d_raw_alloc_zero_f32_0(uint32_t cout, float *gbias)
{
    if (1024U * blockIdx.x + threadIdx.x < cout)
        gbias[1024U * blockIdx.x + threadIdx.x] = 0.0f;
}

Kuiper_KB_Conv1DAlloc_conv1d_raw_result
Kuiper_KB_Conv1DAlloc_conv1d_raw_alloc_zero_f32(uint32_t b, uint32_t cin,
                                                uint32_t l_in, uint32_t cout,
                                                uint32_t kk, uint32_t stride,
                                                uint32_t pad, uint32_t dilation,
                                                float *gx, float *gw)
{
    KPR_GUARD(pad <= 2147483647U);
    uint32_t two_pad = 2U * pad;
    KPR_GUARD(l_in <= 4294967295U - two_pad);
    uint32_t padded = l_in + two_pad;
    uint32_t km1 = kk - 1U;
    KPR_GUARD(km1 <= 4294967295U / dilation);
    uint32_t dilated = dilation * km1;
    KPR_GUARD(dilated <= 4294967295U - 1U);
    KPR_GUARD(dilated + 1U <= padded);
    uint32_t l_out =
        Kuiper_KB_Conv1DAlloc_conv1d_out_dim(l_in, kk, stride, dilation, pad);
    KPR_GUARD(cin <= 4294967295U / b);
    uint32_t xy = b * cin;
    KPR_GUARD(l_in <= 4294967295U / xy);
    KRML_HOST_IGNORE(xy * l_in);
    KPR_GUARD(cin <= 4294967295U / cout);
    uint32_t xy0 = cout * cin;
    KPR_GUARD(kk <= 4294967295U / xy0);
    KRML_HOST_IGNORE(xy0 * kk);
    KPR_GUARD(cout <= 4294967295U / b);
    uint32_t xy1 = b * cout;
    KPR_GUARD(l_out <= 4294967295U / xy1);
    uint32_t ylen = xy1 * l_out;
    KPR_GUARD(kk <= 4294967295U / cin);
    KRML_HOST_IGNORE(cin * kk);
    KPR_GUARD(l_out <= 4294967295U / cout);
    KRML_HOST_IGNORE(cout * l_out);
    KPR_GUARD(stride <= 4294967295U / l_out);
    uint32_t ls = l_out * stride;
    KPR_GUARD(dilation <= 4294967295U / kk);
    uint32_t kd = kk * dilation;
    KPR_GUARD(ls <= 4294967295U - kd);
    KRML_HOST_IGNORE(ls + kd);
    KPR_GUARD(ylen <= 2147483648U);
    float *gbias = (float *) KPR_GPU_ALLOC(sizeof(float), cout);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_conv1d_raw_alloc_zero_f32_0,
              cout / 1024U + (uint32_t) (cout % 1024U != 0U), 1024U, 0U, s,
              cout, gbias);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    Kuiper_KB_Conv1DAlloc_conv1d_raw_result r =
        Kuiper_KB_Conv1DAlloc_conv1d_raw_alloc_bias_f32(
            b, cin, l_in, cout, kk, stride, pad, dilation, gx, gw, gbias);
    MUST(cudaFree(gbias));
    return r;
}
