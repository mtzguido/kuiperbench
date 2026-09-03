
#ifndef Kuiper_KB_TripletMarginLoss_H
#define Kuiper_KB_TripletMarginLoss_H

#include <kuiper.h>
#include <kbench.h>

extern float Kuiper_KB_TripletMarginLoss_triplet_default_eps_f32;

float *Kuiper_KB_TripletMarginLoss_triplet_scalar_out_f32(float x);

float *Kuiper_KB_TripletMarginLoss_triplet_fw_f32(uint32_t b, uint32_t d,
                                                  double margin64,
                                                  float *anchor,
                                                  float *positive,
                                                  float *negative);

#define Kuiper_KB_TripletMarginLoss_H_DEFINED
#endif /* Kuiper_KB_TripletMarginLoss_H */
