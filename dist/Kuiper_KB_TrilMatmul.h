
#ifndef Kuiper_KB_TrilMatmul_H
#define Kuiper_KB_TrilMatmul_H

#include <kuiper.h>
#include <kbench.h>

void Kuiper_KB_TrilMatmul_tril_matmul_f32(uint32_t n, float *gA, float *gB,
                                          float *y);

#define Kuiper_KB_TrilMatmul_H_DEFINED
#endif /* Kuiper_KB_TrilMatmul_H */
