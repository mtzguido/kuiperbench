
#ifndef Kuiper_KB_ConvT3DGeneral_H
#define Kuiper_KB_ConvT3DGeneral_H

#include <kuiper.h>
#include <kbench.h>

uint32_t Kuiper_KB_ConvT3DGeneral_convt_out_dim(uint32_t n, uint32_t s,
                                                uint32_t d, uint32_t k,
                                                uint32_t p, uint32_t opad);

void Kuiper_KB_ConvT3DGeneral_convt3d_general_f32(
    uint32_t b, uint32_t cin, uint32_t d_in, uint32_t h_in, uint32_t w_in,
    uint32_t cout, uint32_t kd, uint32_t kh, uint32_t kw, uint32_t sd,
    uint32_t sh, uint32_t sw, uint32_t pd, uint32_t ph, uint32_t pw,
    uint32_t dd, uint32_t dh, uint32_t dw, uint32_t d_out, uint32_t h_out,
    uint32_t w_out, float *gx, float *gw, float *gbias, float *gy);

float *Kuiper_KB_ConvT3DGeneral_convt3d_general_alloc_f32(
    uint32_t b, uint32_t cin, uint32_t d_in, uint32_t h_in, uint32_t w_in,
    uint32_t cout, uint32_t kd, uint32_t kh, uint32_t kw, uint32_t sd,
    uint32_t sh, uint32_t sw, uint32_t pd, uint32_t ph, uint32_t pw,
    uint32_t dd, uint32_t dh, uint32_t dw, uint32_t d_out, uint32_t h_out,
    uint32_t w_out, float *gx, float *gw, float *gbias);

#define Kuiper_KB_ConvT3DGeneral_H_DEFINED
#endif /* Kuiper_KB_ConvT3DGeneral_H */
