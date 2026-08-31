
#ifndef Kuiper_KB_TripletMarginLoss_H
#define Kuiper_KB_TripletMarginLoss_H

#include <kuiper.h>
#include <kbench.h>

float Kuiper_KB_TripletMarginLoss_triplet_fw_f32(uint32_t b, uint32_t d,
                                                 float margin, float eps,
                                                 float *anchor, float *positive,
                                                 float *negative);

#define Kuiper_KB_TripletMarginLoss_H_DEFINED
#endif /* Kuiper_KB_TripletMarginLoss_H */
