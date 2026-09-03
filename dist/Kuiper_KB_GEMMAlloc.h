
#ifndef Kuiper_KB_GEMMAlloc_H
#define Kuiper_KB_GEMMAlloc_H

#include <kuiper.h>
#include <kbench.h>

float *Kuiper_KB_GEMMAlloc_gemm_naive3_alloc_f32(uint32_t m, uint32_t n,
                                                 uint32_t k, float *a,
                                                 float *b);

float *Kuiper_KB_GEMMAlloc_gemm_naive1_alloc_f32(uint32_t m, uint32_t n,
                                                 uint32_t k, float *a,
                                                 float *b);

#define Kuiper_KB_GEMMAlloc_H_DEFINED
#endif /* Kuiper_KB_GEMMAlloc_H */
