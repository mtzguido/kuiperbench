
#ifndef Kuiper_KB_AvgPool2D_H
#define Kuiper_KB_AvgPool2D_H

#include <kuiper.h>
#include <kbench.h>

uint32_t Kuiper_KB_AvgPool2D_pool_out_len_1d_sz(uint32_t l, uint32_t k,
                                                uint32_t s, uint32_t p,
                                                uint32_t d);

float Kuiper_KB_AvgPool2D_avgpool_recip_f32(uint32_t k);

void Kuiper_KB_AvgPool2D_avgpool2d_axis_fw_rm_f32(uint32_t k, uint32_t s,
                                                  uint32_t p, uint32_t d,
                                                  uint32_t bc, uint32_t l,
                                                  uint32_t l_out, float *input,
                                                  float *output);

Prims_dtuple2__uint32_t__float_
Kuiper_KB_AvgPool2D_avgpool2d_axis_alloc_f32(uint32_t k, uint32_t s, uint32_t p,
                                             uint32_t d, uint32_t bc,
                                             uint32_t l, float *input);

#define Kuiper_KB_AvgPool2D_H_DEFINED
#endif /* Kuiper_KB_AvgPool2D_H */
