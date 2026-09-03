
#include "Kuiper_KB_ScalarMul.h"

__global__
/**
  hoisted when extracting smul_fw_f32
*/
static void
__hoisted_smul_fw_f32_0(float c, uint32_t lena, float *a)
{
    if (1024U * blockIdx.x + threadIdx.x < lena)
        a[1024U * blockIdx.x + threadIdx.x] *= c;
}

void Kuiper_KB_ScalarMul_smul_fw_f32(float c, uint32_t lena, float *a)
{
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_smul_fw_f32_0,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U), 1024U, 0U, s1, c,
              lena, a);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
}

__global__
/**
  hoisted when extracting smul_fw_f64
*/
static void
__hoisted_smul_fw_f64_0(double c, uint32_t lena, double *a)
{
    if (1024U * blockIdx.x + threadIdx.x < lena)
        a[1024U * blockIdx.x + threadIdx.x] *= c;
}

void Kuiper_KB_ScalarMul_smul_fw_f64(double c, uint32_t lena, double *a)
{
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_smul_fw_f64_0,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U), 1024U, 0U, s1, c,
              lena, a);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
}

__global__
/**
  hoisted when extracting smul_fw_u32
*/
static void
__hoisted_smul_fw_u32_0(uint32_t c, uint32_t lena, uint32_t *a)
{
    if (1024U * blockIdx.x + threadIdx.x < lena)
        a[1024U * blockIdx.x + threadIdx.x] *= c;
}

void Kuiper_KB_ScalarMul_smul_fw_u32(uint32_t c, uint32_t lena, uint32_t *a)
{
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_smul_fw_u32_0,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U), 1024U, 0U, s1, c,
              lena, a);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
}

__global__
/**
  hoisted when extracting smul_fw_u64
*/
static void
__hoisted_smul_fw_u64_0(uint64_t c, uint32_t lena, uint64_t *a)
{
    if (1024U * blockIdx.x + threadIdx.x < lena)
        a[1024U * blockIdx.x + threadIdx.x] *= c;
}

void Kuiper_KB_ScalarMul_smul_fw_u64(uint64_t c, uint32_t lena, uint64_t *a)
{
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_smul_fw_u64_0,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U), 1024U, 0U, s1, c,
              lena, a);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
}

__global__
/**
  hoisted when extracting smul_out_f32
*/
static void
__hoisted_smul_out_f32_0(float cst, uint32_t lena, float *c, float *a)
{
    if (1024U * blockIdx.x + threadIdx.x < lena)
        c[1024U * blockIdx.x + threadIdx.x] =
            cst * a[1024U * blockIdx.x + threadIdx.x];
}

void Kuiper_KB_ScalarMul_smul_out_f32(float cst, uint32_t lena, float *c,
                                      float *a)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_smul_out_f32_0,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U), 1024U, 0U, s, cst,
              lena, c, a);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting smul_out_f64
*/
static void
__hoisted_smul_out_f64_0(double cst, uint32_t lena, double *c, double *a)
{
    if (1024U * blockIdx.x + threadIdx.x < lena)
        c[1024U * blockIdx.x + threadIdx.x] =
            cst * a[1024U * blockIdx.x + threadIdx.x];
}

void Kuiper_KB_ScalarMul_smul_out_f64(double cst, uint32_t lena, double *c,
                                      double *a)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_smul_out_f64_0,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U), 1024U, 0U, s, cst,
              lena, c, a);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting smul_out_u32
*/
static void
__hoisted_smul_out_u32_0(uint32_t cst, uint32_t lena, uint32_t *c, uint32_t *a)
{
    if (1024U * blockIdx.x + threadIdx.x < lena)
        c[1024U * blockIdx.x + threadIdx.x] =
            cst * a[1024U * blockIdx.x + threadIdx.x];
}

void Kuiper_KB_ScalarMul_smul_out_u32(uint32_t cst, uint32_t lena, uint32_t *c,
                                      uint32_t *a)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_smul_out_u32_0,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U), 1024U, 0U, s, cst,
              lena, c, a);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting smul_out_u64
*/
static void
__hoisted_smul_out_u64_0(uint64_t cst, uint32_t lena, uint64_t *c, uint64_t *a)
{
    if (1024U * blockIdx.x + threadIdx.x < lena)
        c[1024U * blockIdx.x + threadIdx.x] =
            cst * a[1024U * blockIdx.x + threadIdx.x];
}

void Kuiper_KB_ScalarMul_smul_out_u64(uint64_t cst, uint32_t lena, uint64_t *c,
                                      uint64_t *a)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_smul_out_u64_0,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U), 1024U, 0U, s, cst,
              lena, c, a);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

float *Kuiper_KB_ScalarMul_smul_alloc_f32(float cst, uint32_t lena, float *a)
{
    float *c = (float *) KPR_GPU_ALLOC(sizeof(float), lena);
    Kuiper_KB_ScalarMul_smul_out_f32(cst, lena, c, a);
    return c;
}

float *Kuiper_KB_ScalarMul_smul_alloc_f64_f32(double cst, uint32_t lena,
                                              float *a)
{
    return Kuiper_KB_ScalarMul_smul_alloc_f32((float) cst, lena, a);
}
