
#ifndef Kuiper_KB_AvgPool1D_H
#define Kuiper_KB_AvgPool1D_H

#include <kuiper.h>
#include <kbench.h>

typedef struct Kuiper_KB_AvgPool1D_avgpool1d_alloc_result_s {
    uint32_t l_out;
    float *output;
} Kuiper_KB_AvgPool1D_avgpool1d_alloc_result;

uint32_t Kuiper_KB_AvgPool1D_pool_out_len_1d_sz(uint32_t l, uint32_t k,
                                                uint32_t s, uint32_t p,
                                                uint32_t d);

float Kuiper_KB_AvgPool1D_avgpool_recip_f32(uint32_t k);

void Kuiper_KB_AvgPool1D_avgpool1d_fw_rm_f32(uint32_t k, uint32_t s, uint32_t p,
                                             uint32_t d, uint32_t bc,
                                             uint32_t l, uint32_t l_out,
                                             float *input, float *output);

Kuiper_KB_AvgPool1D_avgpool1d_alloc_result
Kuiper_KB_AvgPool1D_avgpool1d_alloc_f32(uint32_t k, uint32_t s, uint32_t p,
                                        uint32_t d, uint32_t bc, uint32_t l,
                                        float *input);

Kuiper_KB_AvgPool1D_avgpool1d_alloc_result
Kuiper_KB_AvgPool1D_avgpool1d_raw_alloc_f32(uint32_t k, uint32_t s, uint32_t p,
                                            uint32_t b, uint32_t c, uint32_t l,
                                            float *input);

#define Kuiper_KB_AvgPool1D_H_DEFINED
#endif /* Kuiper_KB_AvgPool1D_H */
