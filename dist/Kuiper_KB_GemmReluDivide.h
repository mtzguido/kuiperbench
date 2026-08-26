
#ifndef Kuiper_KB_GemmReluDivide_H
#define Kuiper_KB_GemmReluDivide_H

#include <kuiper.h>
#include <kbench.h>

void Kuiper_KB_GemmReluDivide_gemm_relu_divide_f32(uint32_t batch,
                                                   uint32_t input, uint32_t out,
                                                   float divisor, float *x,
                                                   float *wt, float *bias,
                                                   float *y);

#define Kuiper_KB_GemmReluDivide_H_DEFINED
#endif /* Kuiper_KB_GemmReluDivide_H */
