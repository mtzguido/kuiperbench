// L1 #57 bridge: ConvTranspose2D, square input, square kernel.
#include "kuiper_convt2d_general_common.h"
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_convt2d_general", &kuiper_convt2d_general_cuda,
          "Kuiper verified ConvTranspose2D forward (general parameters)");
}
