
#ifndef Kuiper_KB_AvgPool3D_H
#define Kuiper_KB_AvgPool3D_H

#include <kuiper.h>
#include <kbench.h>

typedef struct Kuiper_KB_AvgPool3D_avgpool3d_axis_alloc_result_s {
    uint32_t l_out;
    float *output;
} Kuiper_KB_AvgPool3D_avgpool3d_axis_alloc_result;

uint32_t Kuiper_KB_AvgPool3D_pool_out_len_1d_sz(uint32_t l, uint32_t k,
                                                uint32_t s, uint32_t p,
                                                uint32_t d);

float Kuiper_KB_AvgPool3D_avgpool_recip_f32(uint32_t k);

void Kuiper_KB_AvgPool3D_avgpool3d_axis_fw_rm_f32(uint32_t k, uint32_t s,
                                                  uint32_t p, uint32_t d,
                                                  uint32_t bc, uint32_t l,
                                                  uint32_t l_out, float *input,
                                                  float *output);

Kuiper_KB_AvgPool3D_avgpool3d_axis_alloc_result
Kuiper_KB_AvgPool3D_avgpool3d_axis_alloc_f32(uint32_t k, uint32_t s, uint32_t p,
                                             uint32_t d, uint32_t bc,
                                             uint32_t l, float *input);

typedef struct Kuiper_KB_AvgPool3D_avgpool3d_full_result_s {
    uint32_t w_out;
    uint32_t h_out;
    uint32_t d_out;
    float *full_output;
} Kuiper_KB_AvgPool3D_avgpool3d_full_result;

Kuiper_KB_AvgPool3D_avgpool3d_full_result
Kuiper_KB_AvgPool3D_avgpool3d_full_alloc_f32(
    uint32_t kd, uint32_t kh, uint32_t kw, uint32_t sd, uint32_t sh,
    uint32_t sw, uint32_t pd, uint32_t ph, uint32_t pw, uint32_t dd,
    uint32_t dh, uint32_t dw, uint32_t bc, uint32_t depth, uint32_t h,
    uint32_t w, float *input);

Kuiper_KB_AvgPool3D_avgpool3d_full_result
Kuiper_KB_AvgPool3D_avgpool3d_raw_alloc_f32(uint32_t k, uint32_t s, uint32_t p,
                                            uint32_t b, uint32_t c,
                                            uint32_t depth, uint32_t h,
                                            uint32_t w, float *input);

#define Kuiper_KB_AvgPool3D_H_DEFINED
#endif /* Kuiper_KB_AvgPool3D_H */
