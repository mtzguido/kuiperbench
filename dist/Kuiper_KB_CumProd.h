
#ifndef Kuiper_KB_CumProd_H
#define Kuiper_KB_CumProd_H

#include <kuiper.h>
#include <kbench.h>

void Kuiper_KB_CumProd_cumprod_fw_f32(uint32_t b, uint32_t d, float *input,
                                      float *output);

float *Kuiper_KB_CumProd_cumprod_alloc_f32(uint32_t b, uint32_t d,
                                           float *input);

#define Kuiper_KB_CumProd_H_DEFINED
#endif /* Kuiper_KB_CumProd_H */
