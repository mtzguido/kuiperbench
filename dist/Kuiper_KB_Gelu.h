
#ifndef Kuiper_KB_Gelu_H
#define Kuiper_KB_Gelu_H

#include <kuiper.h>
#include <kbench.h>

void Kuiper_KB_Gelu_gelu_fw_f32(uint32_t lena, float *a);

void Kuiper_KB_Gelu_gelu_fw_f64(uint32_t lena, double *a);

#define Kuiper_KB_Gelu_H_DEFINED
#endif /* Kuiper_KB_Gelu_H */
