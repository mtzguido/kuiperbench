
#ifndef Kuiper_KB_Selu_H
#define Kuiper_KB_Selu_H

#include <kuiper.h>
#include <kbench.h>

void Kuiper_KB_Selu_selu_fw_f32(uint32_t lena, float *a);

void Kuiper_KB_Selu_selu_fw_f64(uint32_t lena, double *a);

#define Kuiper_KB_Selu_H_DEFINED
#endif /* Kuiper_KB_Selu_H */
