
#ifndef Kuiper_KB_Gelu_H
#define Kuiper_KB_Gelu_H

#include <kuiper.h>
#include <kbench.h>

void Kuiper_KB_Gelu_gelu_fw_f32(uint32_t lena, float *a);

void Kuiper_KB_Gelu_gelu_fw_f64(uint32_t lena, double *a);

float *Kuiper_KB_Gelu_gelu_alloc_f32(uint32_t lena, float *input);

double *Kuiper_KB_Gelu_gelu_alloc_f64(uint32_t lena, double *input);

#define Kuiper_KB_Gelu_H_DEFINED
#endif /* Kuiper_KB_Gelu_H */
