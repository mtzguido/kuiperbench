
#ifndef Kuiper_KB_HingeLoss_H
#define Kuiper_KB_HingeLoss_H

#include <kuiper.h>
#include <kbench.h>

float *Kuiper_KB_HingeLoss_hinge_loss_broadcast_f32(uint32_t b, uint32_t n,
                                                    float *predictions,
                                                    float *targets);

#define Kuiper_KB_HingeLoss_H_DEFINED
#endif /* Kuiper_KB_HingeLoss_H */
