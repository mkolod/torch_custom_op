import torch
import saxpy_cuda

NUMEL = 1024
dev = 'cuda:0'
x = torch.randn(NUMEL, device=dev);
y = torch.randn(NUMEL, device=dev);
a = 2.0

out_torch = a * x + y
out_custom = saxpy_cuda.forward(x, y, a)

result = torch.allclose(out_torch, out_custom)
print(f"all outputs close? {result}")
assert result, "outputs don't match"
