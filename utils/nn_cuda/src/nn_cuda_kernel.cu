#include "cuda_common.h"
#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <queue>

using namespace std;

template <typename scalar_t>
__global__ void cuNearestNeighborKernel(
    const scalar_t* __restrict__ pred,
    const const scalar_t* __restrict__ embedding, 
    scalar_t* __restrict__ output, 
    size_t batch_size, size_t height, size_t width,
    size_t num_embedding, size_t size_embedding) {

    int hi = threadIdx.x + blockIdx.x*blockDim.x;
    int wi = threadIdx.y + blockIdx.y*blockDim.y;
    if(hi>=batch_size*height||wi>=width) return;

    const size_t index = (hi * width) + wi;
    const size_t s1 = index * size_embedding;

    scalar_t min_val = 1000000000000000;
    size_t min_index = -1;
    for(int i=0; i<num_embedding; ++i) {
        scalar_t sum = 0;
        const size_t s2 = i * size_embedding;
        for(int j=0; j<size_embedding; ++j) {
//          sum += (pred[s1 + j] - embedding[s2 + j]) * (pred[s1 + j] - embedding[s2 + j]);
          sum += - pred[s1 + j] * embedding[s2 + j];
        }
        if(min_val > sum) {
          min_index = i;
          min_val = sum;
        }
    }
    output[index] = min_index;
}

int cuNearestNeighborLaucher(const at::Tensor pred, const at::Tensor embedding,
                          at::Tensor output) {
    const size_t batch_size = pred.size(0);
    const size_t height = pred.size(1);
    const size_t width = pred.size(2);

    const size_t num_embedding = embedding.size(0);
    const size_t size_embedding = embedding.size(1);

    AT_DISPATCH_FLOATING_TYPES(
        pred.type(), "NearestNeighborKernel_cuda", ([&] {
            const scalar_t *pred_data = pred.data<scalar_t>();
            const scalar_t *embedding_data = embedding.data<scalar_t>();
            scalar_t *output_data = output.data<scalar_t>();

            int bdim0, bdim1, bdim2;
            int tdim0, tdim1, tdim2;

            getGPULayout(batch_size * height, width, 1, &bdim0, &bdim1, &bdim2, &tdim0, &tdim1, &tdim2);

            dim3 bdim(bdim0, bdim1, bdim2);
            dim3 tdim(tdim0, tdim1, tdim2);

            cuNearestNeighborKernel<scalar_t>
                <<< bdim, tdim>>>(
                    pred_data, embedding_data, output_data,
                    batch_size, height, width,
                    num_embedding, size_embedding
                    );
        }));
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
    return 1;
}
