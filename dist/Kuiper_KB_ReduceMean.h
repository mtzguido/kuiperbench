
#ifndef Kuiper_KB_ReduceMean_H
#define Kuiper_KB_ReduceMean_H

#include <kuiper.h>
#include <kbench.h>

float Kuiper_KB_ReduceMean_reducemean_recip_f32(uint32_t d);

void Kuiper_KB_ReduceMean_reduce_mean_fw_f32(uint32_t b, uint32_t m, uint32_t d,
                                             float inv_d, float *x, float *y);

#define Kuiper_KB_ReduceMean_H_DEFINED
#endif /* Kuiper_KB_ReduceMean_H */
