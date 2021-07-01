#include <torch/torch.h>

#include <cmath>
#include <vector>

int cuNearestNeighborLaucher(const at::Tensor pred, const at::Tensor embedding,
                          at::Tensor output);

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor nn_cuda(at::Tensor pred, at::Tensor embedding, at::Tensor output) {
  CHECK_INPUT(pred);
  CHECK_INPUT(embedding);
  CHECK_INPUT(output);

  cuNearestNeighborLaucher(pred, embedding, output);

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("NearestNeighbor", &nn_cuda, "NearestNeighbor Laucher (CUDA)");
}
