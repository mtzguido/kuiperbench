
#ifndef Kuiper_KB_KLDivLoss_H
#define Kuiper_KB_KLDivLoss_H

#include <kuiper.h>
#include <kbench.h>

float Kuiper_KB_KLDivLoss_kl_div_mean_f32(float s, uint32_t b);

float Kuiper_KB_KLDivLoss_kl_div_fw_f32(uint32_t n, float *predictions,
                                        float *targets);

#define Kuiper_KB_KLDivLoss_H_DEFINED
#endif /* Kuiper_KB_KLDivLoss_H */
