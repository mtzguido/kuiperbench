
#ifndef Kuiper_KB_RMSNorm_H
#define Kuiper_KB_RMSNorm_H

#include <kuiper.h>
#include <kbench.h>

void Kuiper_KB_RMSNorm_rmsnorm_fw(uint32_t b, uint32_t hw, uint32_t c,
                                  float eps, float *x);

extern void (*Kuiper_KB_RMSNorm_rmsnorm_fw_f32)(uint32_t x0, uint32_t x1,
                                                uint32_t x2, float x3,
                                                float *x4);

float *Kuiper_KB_RMSNorm_rmsnorm4d_alloc_f32(uint32_t b, uint32_t c, uint32_t h,
                                             uint32_t w, double eps, float *x);

#define Kuiper_KB_RMSNorm_H_DEFINED
#endif /* Kuiper_KB_RMSNorm_H */
