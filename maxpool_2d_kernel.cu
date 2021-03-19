#include <torch/extension.h>

#include "maxpool_2d_kernel.h"

__global__ void maxpool2d_kernel(
    const int X_H,
    const int X_W,
    const int Y_H,
    const int Y_W,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_t,
    const int pad_l,
    const float* const __restrict__ X,
    float* const __restrict__ Y) {

  const int X_HxW = X_H * X_W;
  const int Y_HxW = Y_H * Y_W;
  const int nc = blockIdx.x / Y_H;
  const int yh = blockIdx.x % Y_H;
  const float* X_ptr = X + nc * X_HxW;
  float* Y_ptr = Y + nc * Y_HxW;
  const int xh = yh * stride_h;
  const int t = max(xh - pad_t, 0);
  const int b = min(xh - pad_t + kernel_h, X_H);
  for (int yw = threadIdx.x; yw < Y_W; yw += blockDim.x) {
    const int xw = yw * stride_w;
    const int l = max(xw - pad_l, 0);
    const int r = min(xw - pad_l + kernel_w, X_W);
    float val = std::numeric_limits<float>::lowest();
    for (int i = t; i < b; ++i) {
      for (int j = l; j < r; ++j) {
#if __CUDA_ARCH__ >=350
        val = max(val, __ldg(X_ptr + i * X_W + j));
#else
        val = max(val, X_ptr[i * X_W + j]);
#endif
      }
    }
    Y_ptr[yh * Y_W + yw] = val;
  }
}

void maxpool2d_launcher(torch::Tensor input, torch::Tensor output, int height, int width, int outH, int outW, int kernel_size, int stride, int padding) {

  const dim3 threads(32, 32);
  const dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);

  maxpool2d_kernel<<<blocks, threads>>>(height, width,
    outH, outW, kernel_size, kernel_size, stride, stride, padding, padding,
    input.data_ptr<float>(), output.data_ptr<float>());
  
  // check CUDA errors

}


