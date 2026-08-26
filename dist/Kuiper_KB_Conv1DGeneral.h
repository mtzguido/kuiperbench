
#ifndef Kuiper_KB_Conv1DGeneral_H
#define Kuiper_KB_Conv1DGeneral_H

#include <kuiper.h>
#include <kbench.h>

void Kuiper_KB_Conv1DGeneral_conv1d_general_f32(
    uint32_t b, uint32_t cin, uint32_t l_in, uint32_t cout, uint32_t kk,
    uint32_t stride, uint32_t pad, uint32_t dilation, uint32_t l_out, float *gx,
    float *gw, float *gbias, float *gy);

#define Kuiper_KB_Conv1DGeneral_H_DEFINED
#endif /* Kuiper_KB_Conv1DGeneral_H */
