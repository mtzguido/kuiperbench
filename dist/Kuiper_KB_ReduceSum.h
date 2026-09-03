
#ifndef Kuiper_KB_ReduceSum_H
#define Kuiper_KB_ReduceSum_H

#include <kuiper.h>
#include <kbench.h>

void Kuiper_KB_ReduceSum_reduce_sum_fw_f32(uint32_t b, uint32_t m, uint32_t d,
                                           float *x, float *y);

float *Kuiper_KB_ReduceSum_reduce_sum_alloc_f32(uint32_t b, uint32_t m,
                                                uint32_t d, float *x);

#define Kuiper_KB_ReduceSum_H_DEFINED
#endif /* Kuiper_KB_ReduceSum_H */
