
#ifndef Kuiper_KB_ScalarMul_H
#define Kuiper_KB_ScalarMul_H

#include <kuiper.h>
#include <kbench.h>

void Kuiper_KB_ScalarMul_smul_fw_f32(float c, uint32_t lena, float *a);

void Kuiper_KB_ScalarMul_smul_fw_f64(double c, uint32_t lena, double *a);

void Kuiper_KB_ScalarMul_smul_fw_u32(uint32_t c, uint32_t lena, uint32_t *a);

void Kuiper_KB_ScalarMul_smul_fw_u64(uint64_t c, uint32_t lena, uint64_t *a);

void Kuiper_KB_ScalarMul_smul_out_f32(float cst, uint32_t lena, float *c,
                                      float *a);

void Kuiper_KB_ScalarMul_smul_out_f64(double cst, uint32_t lena, double *c,
                                      double *a);

void Kuiper_KB_ScalarMul_smul_out_u32(uint32_t cst, uint32_t lena, uint32_t *c,
                                      uint32_t *a);

void Kuiper_KB_ScalarMul_smul_out_u64(uint64_t cst, uint32_t lena, uint64_t *c,
                                      uint64_t *a);

float *Kuiper_KB_ScalarMul_smul_alloc_f32(float cst, uint32_t lena, float *a);

float *Kuiper_KB_ScalarMul_smul_alloc_f64_f32(double cst, uint32_t lena,
                                              float *a);

#define Kuiper_KB_ScalarMul_H_DEFINED
#endif /* Kuiper_KB_ScalarMul_H */
