
#ifndef Kuiper_KB_ArgminReduceDim_H
#define Kuiper_KB_ArgminReduceDim_H

#include <kuiper.h>
#include <kbench.h>

void Kuiper_KB_ArgminReduceDim_argminreduce_dim_fw_f32(uint32_t b, uint32_t m,
                                                       uint32_t d, float *x,
                                                       int64_t *y);

int64_t *Kuiper_KB_ArgminReduceDim_argminreduce_dim_alloc_f32(uint32_t b,
                                                              uint32_t m,
                                                              uint32_t d,
                                                              float *x);

#define Kuiper_KB_ArgminReduceDim_H_DEFINED
#endif /* Kuiper_KB_ArgminReduceDim_H */
