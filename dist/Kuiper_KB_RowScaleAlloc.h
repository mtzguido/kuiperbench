
#ifndef Kuiper_KB_RowScaleAlloc_H
#define Kuiper_KB_RowScaleAlloc_H

#include <kuiper.h>
#include <kbench.h>

float *Kuiper_KB_RowScaleAlloc_row_scale_alloc_f32(uint32_t m, uint32_t n,
                                                   float *a, float *b);

#define Kuiper_KB_RowScaleAlloc_H_DEFINED
#endif /* Kuiper_KB_RowScaleAlloc_H */
