
#ifndef Kuiper_KB_SDPA_H
#define Kuiper_KB_SDPA_H

#include <kuiper.h>
#include <kbench.h>

float *Kuiper_KB_SDPA_sdpa_f32(uint32_t b, uint32_t h, uint32_t s, uint32_t d,
                               float *gQ, float *gK, float *gV);

#define Kuiper_KB_SDPA_H_DEFINED
#endif /* Kuiper_KB_SDPA_H */
