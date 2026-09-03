
#ifndef Kuiper_KB_HuberLoss_H
#define Kuiper_KB_HuberLoss_H

#include <kuiper.h>
#include <kbench.h>

float *Kuiper_KB_HuberLoss_huber_scalar_out_f32(float x);

float *Kuiper_KB_HuberLoss_huber_loss_fw_f32(uint32_t n, float *predictions,
                                             float *targets);

#define Kuiper_KB_HuberLoss_H_DEFINED
#endif /* Kuiper_KB_HuberLoss_H */
