from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='ransac_voting_gpu',
    ext_modules=[
        CUDAExtension('ransac_voting_gpu', [
            './src/ransac_voting_gpu.cpp',
            './src/ransac_voting_kernel.cu'
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
