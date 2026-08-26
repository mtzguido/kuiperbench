
#ifndef Kuiper_KB_MaxPool1D_H
#define Kuiper_KB_MaxPool1D_H

#include <kuiper.h>
#include <kbench.h>

uint32_t Kuiper_KB_MaxPool1D_pool_out_len_1d_sz(uint32_t l, uint32_t k,
                                                uint32_t s, uint32_t p,
                                                uint32_t d);

void Kuiper_KB_MaxPool1D_maxpool1d_fw_rm_f32(uint32_t k, uint32_t s, uint32_t p,
                                             uint32_t d, uint32_t bc,
                                             uint32_t l, uint32_t l_out,
                                             float *input, float *output);

Prims_dtuple2__uint32_t__float_
Kuiper_KB_MaxPool1D_maxpool1d_alloc_f32(uint32_t b, uint32_t c, uint32_t l,
                                        uint32_t k, uint32_t s, uint32_t p,
                                        uint32_t d, float *input);

#define Kuiper_KB_MaxPool1D_H_DEFINED
#endif /* Kuiper_KB_MaxPool1D_H */
