import torch
import saxpy_cuda

NUMEL = 1024
x = torch.randn(NUMEL, device='cuda:0');
y = torch.randn(NUMEL, device='cuda:0');
a = 2.0

out_torch = a * x + y
out_custom = saxpy_cuda.forward(x, y, a)

print("all outputs close? {}".format(torch.allclose(out_torch, out_custom)))
