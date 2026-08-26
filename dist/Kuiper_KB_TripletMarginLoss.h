
#ifndef Kuiper_KB_TripletMarginLoss_H
#define Kuiper_KB_TripletMarginLoss_H

#include <kuiper.h>
#include <kbench.h>

float Kuiper_KB_TripletMarginLoss_triplet_recip_f32(uint32_t b);

float Kuiper_KB_TripletMarginLoss_sq_diff_step_f32(float x, float y);

float Kuiper_KB_TripletMarginLoss_triplet_step_f32(float margin, float d_ap,
                                                   float d_an);

float Kuiper_KB_TripletMarginLoss_triplet_fw_f32(uint32_t b, uint32_t d,
                                                 float margin, float inv_b,
                                                 float *a, float *p, float *n);

#define Kuiper_KB_TripletMarginLoss_H_DEFINED
#endif /* Kuiper_KB_TripletMarginLoss_H */
