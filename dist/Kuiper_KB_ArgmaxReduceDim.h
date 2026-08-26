
#ifndef Kuiper_KB_ArgmaxReduceDim_H
#define Kuiper_KB_ArgmaxReduceDim_H

#include <kuiper.h>
#include <kbench.h>

void Kuiper_KB_ArgmaxReduceDim_argmaxreduce_dim_fw_f32(uint32_t b, uint32_t m,
                                                       uint32_t d, float *x,
                                                       int64_t *y);

#define Kuiper_KB_ArgmaxReduceDim_H_DEFINED
#endif /* Kuiper_KB_ArgmaxReduceDim_H */
