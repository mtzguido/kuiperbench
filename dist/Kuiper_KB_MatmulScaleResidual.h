
#ifndef Kuiper_KB_MatmulScaleResidual_H
#define Kuiper_KB_MatmulScaleResidual_H

#include <kuiper.h>
#include <kbench.h>

void Kuiper_KB_MatmulScaleResidual_matmul_scale_residual_f32(
    uint32_t batch, uint32_t input, uint32_t out, float sf, float *x, float *wt,
    float *bias, float *y);

#define Kuiper_KB_MatmulScaleResidual_H_DEFINED
#endif /* Kuiper_KB_MatmulScaleResidual_H */
