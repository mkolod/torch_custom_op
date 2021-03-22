#pragma once
#include <torch/extension.h>

void saxpy_launcher(torch::Tensor* x,  torch::Tensor* y, torch::Tensor* output,  int numel,  float a);
