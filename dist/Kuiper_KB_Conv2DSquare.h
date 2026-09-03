
#ifndef Kuiper_KB_Conv2DSquare_H
#define Kuiper_KB_Conv2DSquare_H

#include <kuiper.h>
#include <kbench.h>

uint32_t Kuiper_KB_Conv2DSquare_conv2d_square_out_sz(uint32_t l, uint32_t k);

void Kuiper_KB_Conv2DSquare_conv2d_square_f32(uint32_t b, uint32_t cin,
                                              uint32_t h_in, uint32_t cout,
                                              uint32_t k, uint32_t h_out,
                                              float *gx, float *gw,
                                              float *gbias, float *gy);

float *Kuiper_KB_Conv2DSquare_conv2d_square63_alloc_f32(float *gx, float *gw);

#define Kuiper_KB_Conv2DSquare_H_DEFINED
#endif /* Kuiper_KB_Conv2DSquare_H */
