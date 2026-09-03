
#ifndef Kuiper_KB_Conv2DDilatedAsym_H
#define Kuiper_KB_Conv2DDilatedAsym_H

#include <kuiper.h>
#include <kbench.h>

uint32_t Kuiper_KB_Conv2DDilatedAsym_conv2dd_out_dim_sz(uint32_t l, uint32_t k,
                                                        uint32_t s, uint32_t d,
                                                        uint32_t p);

void Kuiper_KB_Conv2DDilatedAsym_conv2d_dilated_asym_f32(
    uint32_t b, uint32_t cin, uint32_t h_in, uint32_t w_in, uint32_t cout,
    uint32_t kh, uint32_t kw, uint32_t sh, uint32_t sw, uint32_t ph,
    uint32_t pw, uint32_t dh, uint32_t dw, uint32_t h_out, uint32_t w_out,
    float *gx, float *gw, float *gbias, float *gy);

float *Kuiper_KB_Conv2DDilatedAsym_conv2d_dilated_asym80_alloc_f32(float *gx,
                                                                   float *gw);

#define Kuiper_KB_Conv2DDilatedAsym_H_DEFINED
#endif /* Kuiper_KB_Conv2DDilatedAsym_H */
