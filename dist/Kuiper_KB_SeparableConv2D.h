
#ifndef Kuiper_KB_SeparableConv2D_H
#define Kuiper_KB_SeparableConv2D_H

#include <kuiper.h>
#include <kbench.h>

uint32_t Kuiper_KB_SeparableConv2D_separable_out_dim(uint32_t n, uint32_t k,
                                                     uint32_t stride,
                                                     uint32_t pad);

float *Kuiper_KB_SeparableConv2D_separable_alloc_f32(
    uint32_t b, uint32_t c, uint32_t h_in, uint32_t w_in, uint32_t kh,
    uint32_t kw, uint32_t stride, uint32_t pad, uint32_t cout, uint32_t h_out,
    uint32_t w_out, float *gx, float *gw_dw, float *gbias_dw, float *gw_pw,
    float *gbias_pw);

float *Kuiper_KB_SeparableConv2D_separable86_alloc_f32(float *gx, float *gw_dw,
                                                       float *gw_pw);

#define Kuiper_KB_SeparableConv2D_H_DEFINED
#endif /* Kuiper_KB_SeparableConv2D_H */
