// L1 #67 bridge: standard 1D conv (groups=1, configurable params).
#include "kuiper_conv1d_general_common.h"
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_conv1d_general", &kuiper_conv1d_general_cuda,
          "Kuiper verified Conv1D forward (general parameters)");
}
