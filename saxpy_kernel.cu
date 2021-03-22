#include "saxpy_kernel.h"
#include <cstdio>

//Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   exit(0); \
 }                                                                 \
}

__global__ void saxpy_kernel(
  const float* const __restrict__ x,
  const float* const __restrict__ y,
  float* const __restrict__ output,
  const int numel,
  const float a) {

  const int gIdx = blockDim.x * blockIdx.x + threadIdx.x;
  if (gIdx > numel - 1) {
    return;
  }

  output[gIdx] = a * x[gIdx] + y[gIdx];
}

void saxpy_launcher(torch::Tensor* x,torch::Tensor* y, torch::Tensor* output, const int numel, const float a) {

  const int block = 1024;
  const int grid = (numel + block - 1) / block; 

  saxpy_kernel<<<grid, block>>>(x->data_ptr<float>(), y->data_ptr<float>(), output->data_ptr<float>(), numel, a);

   cudaStreamSynchronize(0); 
   cudaCheckError();
}


