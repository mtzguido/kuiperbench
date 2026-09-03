
#ifndef Kuiper_KB_MeanVarNorm_H
#define Kuiper_KB_MeanVarNorm_H

#include <kuiper.h>
#include <kbench.h>

void Kuiper_KB_MeanVarNorm_mean_var_norm(uint32_t b, uint32_t d, float eps,
                                         float inv_d, float *x);

void Kuiper_KB_MeanVarNorm_mean_var_norm_fw(uint32_t b, uint32_t d, float eps,
                                            float *x);

extern void (*Kuiper_KB_MeanVarNorm_mean_var_norm_fw_f32)(uint32_t x0,
                                                          uint32_t x1, float x2,
                                                          float *x3);

float *Kuiper_KB_MeanVarNorm_instancenorm34_alloc_f32(uint32_t b, uint32_t c,
                                                      uint32_t h, uint32_t w,
                                                      double eps, float *x);

float *Kuiper_KB_MeanVarNorm_groupnorm35_alloc_f32(uint32_t b, uint32_t c,
                                                   uint32_t h, uint32_t w,
                                                   uint32_t groups, double eps,
                                                   float *x);

#define Kuiper_KB_MeanVarNorm_H_DEFINED
#endif /* Kuiper_KB_MeanVarNorm_H */
