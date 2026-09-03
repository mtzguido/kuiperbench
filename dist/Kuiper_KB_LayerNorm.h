
#ifndef Kuiper_KB_LayerNorm_H
#define Kuiper_KB_LayerNorm_H

#include <kuiper.h>
#include <kbench.h>

void Kuiper_KB_LayerNorm_layer_norm(uint32_t b, uint32_t n, float eps,
                                    float inv_n, float *x, float *gamma,
                                    float *beta);

void Kuiper_KB_LayerNorm_layernorm_fw(uint32_t b, uint32_t n, float eps,
                                      float *x, float *gamma, float *beta);

extern void (*Kuiper_KB_LayerNorm_layernorm_fw_f32)(uint32_t x0, uint32_t x1,
                                                    float x2, float *x3,
                                                    float *x4, float *x5);

float *Kuiper_KB_LayerNorm_layernorm4d_alloc_f32(uint32_t b, uint32_t c,
                                                 uint32_t h, uint32_t w,
                                                 double eps, float *x,
                                                 float *gamma, float *beta);

#define Kuiper_KB_LayerNorm_H_DEFINED
#endif /* Kuiper_KB_LayerNorm_H */
