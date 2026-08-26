
#ifndef Kuiper_KB_LeakyReLU_H
#define Kuiper_KB_LeakyReLU_H

#include <kuiper.h>
#include <kbench.h>

void Kuiper_KB_LeakyReLU_leaky_relu_fw_f32(float slope, uint32_t lena,
                                           float *a);

void Kuiper_KB_LeakyReLU_leaky_relu_fw_f64(double slope, uint32_t lena,
                                           double *a);

#define Kuiper_KB_LeakyReLU_H_DEFINED
#endif /* Kuiper_KB_LeakyReLU_H */
