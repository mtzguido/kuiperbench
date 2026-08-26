
#ifndef Kuiper_KB_MatmulSubMulRelu_H
#define Kuiper_KB_MatmulSubMulRelu_H

#include <kuiper.h>
#include <kbench.h>

void Kuiper_KB_MatmulSubMulRelu_matmul_sub_mul_relu_f32(
    uint32_t batch, uint32_t input, uint32_t out, float sub_v, float mul_v,
    float *x, float *wt, float *bias, float *y);

#define Kuiper_KB_MatmulSubMulRelu_H_DEFINED
#endif /* Kuiper_KB_MatmulSubMulRelu_H */
