
#ifndef Kuiper_KB_MaskedCumSum_H
#define Kuiper_KB_MaskedCumSum_H

#include <kuiper.h>
#include <kbench.h>

float *Kuiper_KB_MaskedCumSum_masked_cumsum_fw_f32(uint32_t b, uint32_t d,
                                                   float *input, uint8_t *mask);

#define Kuiper_KB_MaskedCumSum_H_DEFINED
#endif /* Kuiper_KB_MaskedCumSum_H */
