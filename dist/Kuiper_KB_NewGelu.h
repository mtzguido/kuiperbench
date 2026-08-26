
#ifndef Kuiper_KB_NewGelu_H
#define Kuiper_KB_NewGelu_H

#include <kuiper.h>
#include <kbench.h>

void Kuiper_KB_NewGelu_newgelu_fw_f32(float half, float c, float k,
                                      uint32_t lena, float *a);

void Kuiper_KB_NewGelu_newgelu_fw_f64(double half, double c, double k,
                                      uint32_t lena, double *a);

#define Kuiper_KB_NewGelu_H_DEFINED
#endif /* Kuiper_KB_NewGelu_H */
