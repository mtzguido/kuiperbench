
#ifndef Kuiper_KB_Matmul4D_H
#define Kuiper_KB_Matmul4D_H

#include <kuiper.h>
#include <kbench.h>

void Kuiper_KB_Matmul4D_matmul4d_f32(uint32_t b, uint32_t i, uint32_t j,
                                     uint32_t l, uint32_t k, float *gA,
                                     float *gB, float *gC);

#define Kuiper_KB_Matmul4D_H_DEFINED
#endif /* Kuiper_KB_Matmul4D_H */
