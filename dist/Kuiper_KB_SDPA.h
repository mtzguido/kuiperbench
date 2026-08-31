
#ifndef Kuiper_KB_SDPA_H
#define Kuiper_KB_SDPA_H

#include <kuiper.h>
#include <kbench.h>

void Kuiper_KB_SDPA_sdpa_f32(uint32_t bh, uint32_t s, uint32_t d, float *gQ,
                             float *gKT, float *gV, float *gScores,
                             float *gOut);

#define Kuiper_KB_SDPA_H_DEFINED
#endif /* Kuiper_KB_SDPA_H */
