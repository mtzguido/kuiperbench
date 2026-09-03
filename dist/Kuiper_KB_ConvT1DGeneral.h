
#ifndef Kuiper_KB_ConvT1DGeneral_H
#define Kuiper_KB_ConvT1DGeneral_H

#include <kuiper.h>
#include <kbench.h>

Prims_dtuple2__uint32_t__float_
Kuiper_KB_ConvT1DGeneral_convt1d_general_alloc_f32(
    uint32_t b, uint32_t cin, uint32_t l_in, uint32_t cout, uint32_t k,
    uint32_t s, uint32_t p, uint32_t opad, uint32_t d, float *gx, float *gw);

#define Kuiper_KB_ConvT1DGeneral_H_DEFINED
#endif /* Kuiper_KB_ConvT1DGeneral_H */
