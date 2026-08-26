// Bridge: KernelBench L1 #3 — Verified batched matrix multiplication
// The batch loop is IN the verified F* code.
// We call Kuiper_KB_BatchedGEMM_batched_gemm_f32 with GPU pointers for A, B
// (Array3.t) and a caller-provided output buffer C (Array3.t); the kernel
// overwrites every page of C in place. No temp allocation, no D2D copy.
// Zero assume · zero magic · zero admit.

#include <torch/extension.h>
#include "kuiper.h"

// Include the extracted verified code directly (Karamel generates a
// self-contained .cu with the header guard and function declaration).
#include "Kuiper_KB_BatchedGEMM.cu"

torch::Tensor kuiper_batched_gemm_cuda(torch::Tensor A, torch::Tensor B) {
    auto A_c = A.contiguous();
    auto B_c = B.contiguous();

    int64_t batch  = A_c.size(0);
    int64_t rows   = A_c.size(1);
    int64_t shared = A_c.size(2);
    int64_t cols   = B_c.size(2);

    // The verified kernel overwrites every page of the output, so an
    // uninitialized buffer is fine — no zeroing, no temp alloc, no D2D copy.
    auto out = torch::empty({batch, rows, cols}, A_c.options());

    if (batch > 0 && rows > 0 && cols > 0 && shared > 0) {
        // Write directly into the output tensor (the kernel takes `c`).
        Kuiper_KB_BatchedGEMM_batched_gemm_f32(
            (uint32_t)batch, (uint32_t)rows, (uint32_t)shared, (uint32_t)cols,
            A_c.data_ptr<float>(), B_c.data_ptr<float>(), out.data_ptr<float>());
    }

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_batched_gemm_cuda", &kuiper_batched_gemm_cuda,
          "Verified batched matrix multiplication (batch loop in F*)");
}
