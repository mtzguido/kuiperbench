// L1 #70 bridge: ConvTranspose3D.
#include "kuiper_convt3d_general_common.h"
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_convt3d_general", &kuiper_convt3d_general_cuda,
          "Kuiper verified ConvTranspose3D forward (general parameters)");
}
