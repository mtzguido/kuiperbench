// L1 #62 bridge: 2D conv, square input, asymmetric kernel.
#include "kuiper_conv2d_general_common.h"
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_conv2d_general", &kuiper_conv2d_general_cuda,
          "Kuiper verified Conv2D forward (general parameters)");
}
