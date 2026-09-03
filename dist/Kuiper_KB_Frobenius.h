
#ifndef Kuiper_KB_Frobenius_H
#define Kuiper_KB_Frobenius_H

#include <kuiper.h>
#include <kbench.h>

void Kuiper_KB_Frobenius_frobenius_fw_f32(uint32_t lena, float *a);

float *Kuiper_KB_Frobenius_frobenius_alloc_f32(uint32_t lena, float *a);

#define Kuiper_KB_Frobenius_H_DEFINED
#endif /* Kuiper_KB_Frobenius_H */
