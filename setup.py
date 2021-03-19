from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='max_pool_cuda',
    ext_modules=[
        CUDAExtension('max_pool_cuda', [
            'maxpool_2d.cpp',
            'maxpool_2d_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
