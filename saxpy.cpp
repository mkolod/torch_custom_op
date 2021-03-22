#include "saxpy_kernel.h"

torch::Tensor saxpy_forward(
    torch::Tensor x,
    torch::Tensor y,
    const float a) {

  auto output = torch::zeros_like(x);

  saxpy_launcher(x, y, output, x.numel(), a);

  return output;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &saxpy_forward, "saxpy forward");
}
