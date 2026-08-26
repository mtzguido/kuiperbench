
#ifndef Kuiper_KB_MinReduceDim_H
#define Kuiper_KB_MinReduceDim_H

#include <kuiper.h>
#include <kbench.h>

void Kuiper_KB_MinReduceDim_minreduce_dim_fw_f32(uint32_t b, uint32_t m,
                                                 uint32_t d, float *x,
                                                 float *y);

#define Kuiper_KB_MinReduceDim_H_DEFINED
#endif /* Kuiper_KB_MinReduceDim_H */
