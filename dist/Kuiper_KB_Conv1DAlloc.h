
#ifndef Kuiper_KB_Conv1DAlloc_H
#define Kuiper_KB_Conv1DAlloc_H

#include <kuiper.h>
#include <kbench.h>

uint32_t Kuiper_KB_Conv1DAlloc_conv1d_out_dim(uint32_t n, uint32_t k,
                                              uint32_t stride,
                                              uint32_t dilation, uint32_t pad);

float *Kuiper_KB_Conv1DAlloc_conv1d_general_alloc_f32(
    uint32_t b, uint32_t cin, uint32_t l_in, uint32_t cout, uint32_t kk,
    uint32_t stride, uint32_t pad, uint32_t dilation, uint32_t l_out, float *gx,
    float *gw, float *gbias);

#define Kuiper_KB_Conv1DAlloc_H_DEFINED
#endif /* Kuiper_KB_Conv1DAlloc_H */
