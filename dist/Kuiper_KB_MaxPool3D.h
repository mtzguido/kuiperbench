
#ifndef Kuiper_KB_MaxPool3D_H
#define Kuiper_KB_MaxPool3D_H

#include <kuiper.h>
#include <kbench.h>

uint32_t Kuiper_KB_MaxPool3D_pool_out_len_1d_sz(uint32_t l, uint32_t k,
                                                uint32_t s, uint32_t p,
                                                uint32_t d);

void Kuiper_KB_MaxPool3D_maxpool3d_axis_fw_rm_f32(uint32_t k, uint32_t s,
                                                  uint32_t p, uint32_t d,
                                                  uint32_t bc, uint32_t l,
                                                  uint32_t l_out, float *input,
                                                  float *output);

Prims_dtuple2__uint32_t__float_
Kuiper_KB_MaxPool3D_maxpool3d_axis_alloc_f32(uint32_t k, uint32_t s, uint32_t p,
                                             uint32_t d, uint32_t bc,
                                             uint32_t l, float *input);

typedef struct Kuiper_KB_MaxPool3D_maxpool3d_full_result_s {
    uint32_t fst;
    Prims_dtuple2__uint32_t_Prims_dtuple2__uint32_t__float_ snd;
} Kuiper_KB_MaxPool3D_maxpool3d_full_result;

typedef Kuiper_KB_MaxPool3D_maxpool3d_full_result
    Kuiper_KB_MaxPool3D_maxpool3d_raw_result;

Kuiper_KB_MaxPool3D_maxpool3d_full_result
Kuiper_KB_MaxPool3D_maxpool3d_raw_alloc_f32(uint32_t k, uint32_t s, uint32_t p,
                                            uint32_t d, uint32_t b, uint32_t c,
                                            uint32_t depth, uint32_t h,
                                            uint32_t w, float *input);

#define Kuiper_KB_MaxPool3D_H_DEFINED
#endif /* Kuiper_KB_MaxPool3D_H */
