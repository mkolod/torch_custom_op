#pragma once
#include <torch/extension.h>

void maxpool2d_launcher(torch::Tensor input, torch::Tensor output, int height, int width, int outH, int outW, int kernel_size, int stride, int padding);
