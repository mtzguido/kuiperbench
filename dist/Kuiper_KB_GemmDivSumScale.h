
#ifndef Kuiper_KB_GemmDivSumScale_H
#define Kuiper_KB_GemmDivSumScale_H

#include <kuiper.h>
#include <kbench.h>

void Kuiper_KB_GemmDivSumScale_gemm_div_sum_scale_f32(uint32_t batch,
                                                      uint32_t input,
                                                      uint32_t hidden, float k,
                                                      float *x, float *wt,
                                                      float *y);

#define Kuiper_KB_GemmDivSumScale_H_DEFINED
#endif /* Kuiper_KB_GemmDivSumScale_H */
