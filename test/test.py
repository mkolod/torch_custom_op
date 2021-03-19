import torch
import max_pool_cuda

x = torch.randn(1, 224, 224).cuda()
kernel_size = 3
stride = 1
padding = (kernel_size - 1) // 2

y = torch.nn.functional.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=padding)
z = max_pool_cuda.forward(x, kernel_size, stride, padding)

print(y-z)
