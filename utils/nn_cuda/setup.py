from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="nn_cuda",
    ext_modules=[
        CUDAExtension("nn_cuda", [
            "src/nn_cuda.cpp",
            "src/nn_cuda_kernel.cu",
        ])
    ],
    cmdclass={"build_ext": BuildExtension})


# from torch.utils.cpp_extension import load

# nn = load(name="nn", sources=["src/nn_cuda.cpp", "src/nn_kernel.cu"])