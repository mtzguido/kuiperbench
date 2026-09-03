
#ifndef Kuiper_KB_L1Norm_H
#define Kuiper_KB_L1Norm_H

#include <kuiper.h>
#include <kbench.h>

void Kuiper_KB_L1Norm_l1norm_fw(uint32_t b, uint32_t d, float *x);

extern void (*Kuiper_KB_L1Norm_l1norm_fw_f32)(uint32_t x0, uint32_t x1,
                                              float *x2);

float *Kuiper_KB_L1Norm_l1norm_alloc_f32(uint32_t b, uint32_t d, float *x);

#define Kuiper_KB_L1Norm_H_DEFINED
#endif /* Kuiper_KB_L1Norm_H */
