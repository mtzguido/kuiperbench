
#ifndef Kuiper_KB_L2Norm_H
#define Kuiper_KB_L2Norm_H

#include <kuiper.h>
#include <kbench.h>

void Kuiper_KB_L2Norm_l2norm_fw_f32(uint32_t b, uint32_t d, float *x);

float *Kuiper_KB_L2Norm_l2norm_alloc_f32(uint32_t b, uint32_t d, float *x);

#define Kuiper_KB_L2Norm_H_DEFINED
#endif /* Kuiper_KB_L2Norm_H */
