
#ifndef Klas_RowLogSoftmax_H
#define Klas_RowLogSoftmax_H

#include <kuiper.h>
#include <kbench.h>

void Klas_RowLogSoftmax_row_log_softmax_rm_f32(uint32_t m, uint32_t n,
                                               float *a);

void Klas_RowLogSoftmax_row_log_softmax_rm_f64(uint32_t m, uint32_t n,
                                               double *a);

#define Klas_RowLogSoftmax_H_DEFINED
#endif /* Klas_RowLogSoftmax_H */
