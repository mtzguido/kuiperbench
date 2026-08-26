
#ifndef Kuiper_KB_BatchNorm_H
#define Kuiper_KB_BatchNorm_H

#include <kuiper.h>
#include <kbench.h>

void Kuiper_KB_BatchNorm_batchnorm_fw(uint32_t c, uint32_t hw, uint32_t nhw,
                                      float eps, float *x, float *gamma,
                                      float *beta);

extern void (*Kuiper_KB_BatchNorm_batchnorm_fw_f32)(uint32_t x0, uint32_t x1,
                                                    uint32_t x2, float x3,
                                                    float *x4, float *x5,
                                                    float *x6);

#define Kuiper_KB_BatchNorm_H_DEFINED
#endif /* Kuiper_KB_BatchNorm_H */
