
#ifndef Kuiper_KB_SoftmaxAlloc_H
#define Kuiper_KB_SoftmaxAlloc_H

#include <kuiper.h>
#include <kbench.h>

float *Kuiper_KB_SoftmaxAlloc_softmax_alloc_f32(uint32_t rows, uint32_t cols,
                                                float *input);

double *Kuiper_KB_SoftmaxAlloc_softmax_alloc_f64(uint32_t rows, uint32_t cols,
                                                 double *input);

#define Kuiper_KB_SoftmaxAlloc_H_DEFINED
#endif /* Kuiper_KB_SoftmaxAlloc_H */
