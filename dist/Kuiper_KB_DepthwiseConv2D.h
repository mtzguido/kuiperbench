
#ifndef Kuiper_KB_DepthwiseConv2D_H
#define Kuiper_KB_DepthwiseConv2D_H

#include <kuiper.h>
#include <kbench.h>

uint32_t Kuiper_KB_DepthwiseConv2D_dwconv2d_out_dim(uint32_t n, uint32_t k,
                                                    uint32_t stride,
                                                    uint32_t pad);

void Kuiper_KB_DepthwiseConv2D_dwconv2d_f32(
    uint32_t b, uint32_t c, uint32_t h_in, uint32_t w_in, uint32_t kh,
    uint32_t kw, uint32_t stride, uint32_t pad, uint32_t h_out, uint32_t w_out,
    float *gx, float *gw, float *gbias, float *gy);

float *Kuiper_KB_DepthwiseConv2D_dwconv2d_alloc_f32(
    uint32_t b, uint32_t c, uint32_t h_in, uint32_t w_in, uint32_t kh,
    uint32_t kw, uint32_t stride, uint32_t pad, uint32_t h_out, uint32_t w_out,
    float *gx, float *gw, float *gbias);

#define Kuiper_KB_DepthwiseConv2D_H_DEFINED
#endif /* Kuiper_KB_DepthwiseConv2D_H */
