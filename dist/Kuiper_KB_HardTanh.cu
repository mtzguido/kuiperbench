
#include "Kuiper_KB_HardTanh.h"

__global__ __launch_bounds__(1024)
    /**
      hoisted when extracting htanh_fw_f32
    */
    static void __hoisted_htanh_fw_f32_0(uint32_t lena, float *a)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        float x = a[1024U * blockIdx.x + threadIdx.x];
        a[1024U * blockIdx.x + threadIdx.x] = (float) 1LL < x ? (float) 1LL
                                              : x < (float) 0LL - (float) 1LL
                                                  ? (float) 0LL - (float) 1LL
                                                  : x;
    }
}

void Kuiper_KB_HardTanh_htanh_fw_f32(uint32_t lena, float *a)
{
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_htanh_fw_f32_0,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U), 1024U, 0U, s1,
              lena, a);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
}

__global__ __launch_bounds__(1024)
    /**
      hoisted when extracting htanh_fw_f64
    */
    static void __hoisted_htanh_fw_f64_0(uint32_t lena, double *a)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        double x = a[1024U * blockIdx.x + threadIdx.x];
        a[1024U * blockIdx.x + threadIdx.x] = (double) 1LL < x ? (double) 1LL
                                              : x < (double) 0LL - (double) 1LL
                                                  ? (double) 0LL - (double) 1LL
                                                  : x;
    }
}

void Kuiper_KB_HardTanh_htanh_fw_f64(uint32_t lena, double *a)
{
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_htanh_fw_f64_0,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U), 1024U, 0U, s1,
              lena, a);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
}

__global__ __launch_bounds__(1024)
    /**
      hoisted when extracting htanh_alloc_f32
    */
    static void __hoisted_htanh_alloc_f32_0(uint32_t lena, float *input,
                                            float *output)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        float y = input[1024U * blockIdx.x + threadIdx.x];
        output[1024U * blockIdx.x + threadIdx.x] =
            (float) 1LL < y                 ? (float) 1LL
            : y < (float) 0LL - (float) 1LL ? (float) 0LL - (float) 1LL
                                            : y;
    }
}

float *Kuiper_KB_HardTanh_htanh_alloc_f32(uint32_t lena, float *input)
{
    float *_return;
    bool _return1 = false;
    float *output = (float *) KPR_GPU_ALLOC(sizeof(float), lena);
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_htanh_alloc_f32_0,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U), 1024U, 0U, s1,
              lena, input, output);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
    _return = output;
    _return1 = true;
    return _return;
}

__global__ __launch_bounds__(1024)
    /**
      hoisted when extracting htanh_alloc_f64
    */
    static void __hoisted_htanh_alloc_f64_0(uint32_t lena, double *input,
                                            double *output)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        double y = input[1024U * blockIdx.x + threadIdx.x];
        output[1024U * blockIdx.x + threadIdx.x] =
            (double) 1LL < y                  ? (double) 1LL
            : y < (double) 0LL - (double) 1LL ? (double) 0LL - (double) 1LL
                                              : y;
    }
}

double *Kuiper_KB_HardTanh_htanh_alloc_f64(uint32_t lena, double *input)
{
    double *_return;
    bool _return1 = false;
    double *output = (double *) KPR_GPU_ALLOC(sizeof(double), lena);
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_htanh_alloc_f64_0,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U), 1024U, 0U, s1,
              lena, input, output);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
    _return = output;
    _return1 = true;
    return _return;
}
