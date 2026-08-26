
#ifndef Kuiper_KB_GemmAddRelu_H
#define Kuiper_KB_GemmAddRelu_H

#include <kuiper.h>
#include <kbench.h>

void Kuiper_KB_GemmAddRelu_gemm_add_relu_f32(uint32_t batch, uint32_t input,
                                             uint32_t out, float *x, float *wt,
                                             float *bias, float *y);

#define Kuiper_KB_GemmAddRelu_H_DEFINED
#endif /* Kuiper_KB_GemmAddRelu_H */
