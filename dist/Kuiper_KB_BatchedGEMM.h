
#ifndef Kuiper_KB_BatchedGEMM_H
#define Kuiper_KB_BatchedGEMM_H

#include <kuiper.h>
#include <kbench.h>

void Kuiper_KB_BatchedGEMM_batched_gemm_f32(uint32_t batch, uint32_t rows,
                                            uint32_t shared, uint32_t cols,
                                            float *a, float *b, float *c);

float *Kuiper_KB_BatchedGEMM_batched_gemm_alloc_f32(uint32_t batch,
                                                    uint32_t rows,
                                                    uint32_t shared,
                                                    uint32_t cols, float *a,
                                                    float *b);

#define Kuiper_KB_BatchedGEMM_H_DEFINED
#endif /* Kuiper_KB_BatchedGEMM_H */
