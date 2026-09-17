
#ifndef Kuiper_KB_Elu_H
#define Kuiper_KB_Elu_H

#include <kuiper.h>
#include <kbench.h>

void Kuiper_KB_Elu_elu_fw_f32(float alpha, uint32_t lena, float *a);

void Kuiper_KB_Elu_elu_fw_f64(double alpha, uint32_t lena, double *a);

float *Kuiper_KB_Elu_elu_alloc_f32(float alpha, uint32_t lena, float *input);

double *Kuiper_KB_Elu_elu_alloc_f64(double alpha, uint32_t lena, double *input);

float *Kuiper_KB_Elu_elu_alloc_f64_f32(double alpha, uint32_t lena,
                                       float *input);

#define Kuiper_KB_Elu_H_DEFINED
#endif /* Kuiper_KB_Elu_H */
