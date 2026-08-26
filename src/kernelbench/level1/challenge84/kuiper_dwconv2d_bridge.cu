// L1 #82 bridge: depthwise 2D conv, square input, square kernel.
#include "kuiper_dwconv2d_common.h"
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_dwconv2d", &kuiper_dwconv2d_cuda,
          "Kuiper verified depthwise Conv2D forward (groups = in_channels)");
}
