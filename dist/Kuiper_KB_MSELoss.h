
#ifndef Kuiper_KB_MSELoss_H
#define Kuiper_KB_MSELoss_H

#include <kuiper.h>
#include <kbench.h>

float Kuiper_KB_MSELoss_mse_loss_fw_f32(uint32_t n, float *predictions,
                                        float *targets);

#define Kuiper_KB_MSELoss_H_DEFINED
#endif /* Kuiper_KB_MSELoss_H */
