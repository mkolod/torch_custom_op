from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='saxpy_cuda',
    ext_modules=[
        CUDAExtension('saxpy_cuda', [
            'saxpy.cpp',
            'saxpy_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
