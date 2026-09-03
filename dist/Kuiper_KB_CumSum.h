
#ifndef Kuiper_KB_CumSum_H
#define Kuiper_KB_CumSum_H

#include <kuiper.h>
#include <kbench.h>

void Kuiper_KB_CumSum_cumsum_fw_f32(uint32_t b, uint32_t d, float *input,
                                    float *output);

float *Kuiper_KB_CumSum_cumsum_alloc_f32(uint32_t b, uint32_t d, float *input);

#define Kuiper_KB_CumSum_H_DEFINED
#endif /* Kuiper_KB_CumSum_H */
