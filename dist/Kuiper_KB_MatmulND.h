
#ifndef Kuiper_KB_MatmulND_H
#define Kuiper_KB_MatmulND_H

#include <kuiper.h>
#include <kbench.h>

void Kuiper_KB_MatmulND_matmul_nd_f32(uint32_t n, uint32_t m, uint32_t k,
                                      uint32_t l, float *gA, float *gB,
                                      float *gC);

float *Kuiper_KB_MatmulND_matmul_nd_alloc_f32(uint32_t n, uint32_t m,
                                              uint32_t k, uint32_t l, float *gA,
                                              float *gB);

#define Kuiper_KB_MatmulND_H_DEFINED
#endif /* Kuiper_KB_MatmulND_H */
