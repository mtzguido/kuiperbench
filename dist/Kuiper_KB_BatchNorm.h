
#ifndef Kuiper_KB_BatchNorm_H
#define Kuiper_KB_BatchNorm_H

#include <kuiper.h>
#include <kbench.h>

void Kuiper_KB_BatchNorm_batchnorm_fw_f32(uint32_t c, uint32_t hw, uint32_t nhw,
                                          float eps, float *x, float *gamma,
                                          float *beta);

float *Kuiper_KB_BatchNorm_batchnorm2d_alloc_f32(uint32_t n, uint32_t c,
                                                 uint32_t h, uint32_t w,
                                                 double eps, float *x,
                                                 float *gamma, float *beta);

#define Kuiper_KB_BatchNorm_H_DEFINED
#endif /* Kuiper_KB_BatchNorm_H */
