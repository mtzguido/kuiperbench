
#ifndef Kuiper_KB_GemmMulLeakyRelu_H
#define Kuiper_KB_GemmMulLeakyRelu_H

#include <kuiper.h>
#include <kbench.h>

void Kuiper_KB_GemmMulLeakyRelu_gemm_mul_leaky_relu_f32(
    uint32_t batch, uint32_t input, uint32_t out, float mult, float slope,
    float *x, float *wt, float *bias, float *y);

#define Kuiper_KB_GemmMulLeakyRelu_H_DEFINED
#endif /* Kuiper_KB_GemmMulLeakyRelu_H */
