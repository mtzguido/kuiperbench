// L1 #54 bridge: 3D conv, square input, square kernel.
#include "kuiper_conv3d_general_common.h"
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_conv3d_general", &kuiper_conv3d_general_cuda,
          "Kuiper verified Conv3D forward (general parameters)");
}
