#include <torch/extension.h>
#include "maxpool_2d_kernel.h"

torch::Tensor maxpool2d_forward(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding) {

  const auto height = input.size(1);
  const auto width = input.size(2);

  const auto outW = (width - kernel_size + 2 * padding ) / stride + 1;
  const auto outH = (height - kernel_size + 2 * padding ) / stride + 1;

  const auto options =
    torch::TensorOptions()
      .dtype(torch::kFloat32)
      .device(torch::kCUDA, 0)
      .requires_grad(false);

  auto output = torch::zeros({outH, outW}, options);

  maxpool2d_launcher(input, output, height, width, outH, outW, kernel_size, stride, padding);

  return output;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &maxpool2d_forward, "maxpool2d forward");
}
