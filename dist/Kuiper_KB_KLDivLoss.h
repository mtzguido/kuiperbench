
#ifndef Kuiper_KB_KLDivLoss_H
#define Kuiper_KB_KLDivLoss_H

#include <kuiper.h>
#include <kbench.h>

float Kuiper_KB_KLDivLoss_kl_div_fw_f32(uint32_t n, uint32_t batches,
                                        float *predictions, float *targets);

#define Kuiper_KB_KLDivLoss_H_DEFINED
#endif /* Kuiper_KB_KLDivLoss_H */
