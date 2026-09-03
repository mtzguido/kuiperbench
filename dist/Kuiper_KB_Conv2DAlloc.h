
#ifndef Kuiper_KB_Conv2DAlloc_H
#define Kuiper_KB_Conv2DAlloc_H

#include <kuiper.h>
#include <kbench.h>

uint32_t Kuiper_KB_Conv2DAlloc_conv2d_out_dim(uint32_t n, uint32_t k,
                                              uint32_t stride, uint32_t pad);

float *Kuiper_KB_Conv2DAlloc_conv2d_general_alloc_f32(
    uint32_t b, uint32_t cin, uint32_t h_in, uint32_t w_in, uint32_t cout,
    uint32_t kh, uint32_t kw, uint32_t stride, uint32_t pad, uint32_t h_out,
    uint32_t w_out, float *gx, float *gw, float *gbias);

Prims_dtuple2__uint32_t_Prims_dtuple2__uint32_t__float_
Kuiper_KB_Conv2DAlloc_conv2d_raw_alloc_bias_f32(uint32_t b, uint32_t cin,
                                                uint32_t h_in, uint32_t w_in,
                                                uint32_t cout, uint32_t kh,
                                                uint32_t kw, uint32_t stride,
                                                uint32_t pad, float *gx,
                                                float *gw, float *gbias);

Prims_dtuple2__uint32_t_Prims_dtuple2__uint32_t__float_
Kuiper_KB_Conv2DAlloc_conv2d_raw_alloc_zero_f32(uint32_t b, uint32_t cin,
                                                uint32_t h_in, uint32_t w_in,
                                                uint32_t cout, uint32_t kh,
                                                uint32_t kw, uint32_t stride,
                                                uint32_t pad, float *gx,
                                                float *gw);

#define Kuiper_KB_Conv2DAlloc_H_DEFINED
#endif /* Kuiper_KB_Conv2DAlloc_H */
