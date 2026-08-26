
#ifndef Kuiper_KB_Conv3DGeneral_H
#define Kuiper_KB_Conv3DGeneral_H

#include <kuiper.h>
#include <kbench.h>

void Kuiper_KB_Conv3DGeneral_conv3d_general_f32(
    uint32_t b, uint32_t cin, uint32_t d_in, uint32_t h_in, uint32_t w_in,
    uint32_t cout, uint32_t kd, uint32_t kh, uint32_t kw, uint32_t stride,
    uint32_t pad, uint32_t d_out, uint32_t h_out, uint32_t w_out, float *gx,
    float *gw, float *gbias, float *gy);

#define Kuiper_KB_Conv3DGeneral_H_DEFINED
#endif /* Kuiper_KB_Conv3DGeneral_H */
