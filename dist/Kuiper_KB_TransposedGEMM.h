
#ifndef Kuiper_KB_TransposedGEMM_H
#define Kuiper_KB_TransposedGEMM_H

#include <kuiper.h>
#include <kbench.h>

void Kuiper_KB_TransposedGEMM_matmul_f32_atb(uint32_t m, uint32_t n, uint32_t k,
                                             float *gA, float *gB, float *gC);

void Kuiper_KB_TransposedGEMM_matmul_f32_abt(uint32_t m, uint32_t n, uint32_t k,
                                             float *gA, float *gB, float *gC);

void Kuiper_KB_TransposedGEMM_matmul_f32_atbt(uint32_t m, uint32_t n,
                                              uint32_t k, float *gA, float *gB,
                                              float *gC);

#define Kuiper_KB_TransposedGEMM_H_DEFINED
#endif /* Kuiper_KB_TransposedGEMM_H */
