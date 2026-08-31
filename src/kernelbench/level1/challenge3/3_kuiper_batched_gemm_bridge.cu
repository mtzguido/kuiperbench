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
    TORCH_CHECK(A.dim() == 3 && B.dim() == 3,
                "kuiper #3: expected two 3-D tensors");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32 &&
                B.scalar_type() == torch::kFloat32,
                "kuiper #3: expected float32 tensors");
    TORCH_CHECK(A.is_cuda() && B.is_cuda() && A.device() == B.device(),
                "kuiper #3: tensors must be CUDA tensors on the same device");
    TORCH_CHECK(A.size(0) == B.size(0) && A.size(2) == B.size(1),
                "kuiper #3: batch/shared dimensions do not match");
    auto A_c = A.contiguous();
    auto B_c = B.contiguous();

    int64_t batch  = A_c.size(0);
    int64_t rows   = A_c.size(1);
    int64_t shared = A_c.size(2);
    int64_t cols   = B_c.size(2);
    TORCH_CHECK(batch > 0 && rows > 0 && shared > 0 && cols > 0 &&
                batch <= (int64_t)UINT32_MAX &&
                rows <= (int64_t)UINT32_MAX &&
                shared <= (int64_t)UINT32_MAX &&
                cols <= (int64_t)UINT32_MAX,
                "kuiper #3: dimensions exceed the uint32 ABI");
    TORCH_CHECK(batch <= (int64_t)UINT32_MAX / rows,
                "kuiper #3: batch*rows exceeds uint32");
    int64_t batch_rows = batch * rows;
    TORCH_CHECK(batch_rows <= (int64_t)UINT32_MAX / shared &&
                batch_rows <= (int64_t)UINT32_MAX / cols &&
                batch_rows <= ((int64_t)2097152 * 1024) / cols,
                "kuiper #3: A/output size exceeds the verified bounds");
    TORCH_CHECK(batch <= (int64_t)UINT32_MAX / shared,
                "kuiper #3: batch*shared exceeds uint32");
    int64_t batch_shared = batch * shared;
    TORCH_CHECK(batch_shared <= (int64_t)UINT32_MAX / cols,
                "kuiper #3: B size exceeds uint32");

    // The verified kernel overwrites every page of the output, so an
    // uninitialized buffer is fine — no zeroing, no temp alloc, no D2D copy.
    auto out = torch::empty({batch, rows, cols}, A_c.options());

    Kuiper_KB_BatchedGEMM_batched_gemm_f32(
        (uint32_t)batch, (uint32_t)rows, (uint32_t)shared, (uint32_t)cols,
        A_c.data_ptr<float>(), B_c.data_ptr<float>(), out.data_ptr<float>());

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_batched_gemm_cuda", &kuiper_batched_gemm_cuda,
          "Verified batched matrix multiplication (batch loop in F*)");
}
