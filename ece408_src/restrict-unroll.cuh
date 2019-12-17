
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

//OPTIMIZATION: LOOP UNROLLING WITH __restrict__

#define TILE_WIDTH 16

__global__ void forward_kernel(float * __restrict__ y, const float * __restrict__ x, const float * __restrict__ k, const int B, const int M, const int C, const int H, const int W, const int K)
{
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

const int H_out = H - K + 1;
const int W_out = W - K + 1;
int W_grid = W_out / TILE_WIDTH;
if (W_out % TILE_WIDTH) W_grid++;
int H_grid = H_out / TILE_WIDTH;
if (H_out % TILE_WIDTH) H_grid++;
int n, m, h, w;
n = blockIdx.x;
m = blockIdx.y;
h = (blockIdx.z / W_grid)*TILE_WIDTH + threadIdx.y;
w = (blockIdx.z % W_grid)*TILE_WIDTH + threadIdx.x;

if(h < H_out && w < W_out) {

	float sum = 0.0;

	for (int c = 0; c < C; c += 3) {
		#pragma unroll
		for (int p = 0; p < K; p++) {
			#pragma unroll
			for (int q = 0; q < K; q++) {
				sum += x4d(n, c, h + p, w + q) * k4d(m, c, p, q);
				if (c+1 < C)
					sum += x4d(n, c+1, h + p, w + q) * k4d(m, c+1, p, q);
				if(c+2 < C)
					sum += x4d(n, c+2, h + p, w + q) * k4d(m, c+2, p, q);
			}
		}
	}

	y4d(n, m, h, w) = sum;
}


#undef y4d
#undef x4d
#undef k4d
}

template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &k)
{
    // Extract the tensor dimensions into B,M,C,H,W,K
	const int B = x.shape_[0];	// Batch size
    const int M = y.shape_[1];
    const int C = x.shape_[1];
    const int H = x.shape_[2];
    const int W = x.shape_[3];
    const int K = k.shape_[3];
	const int H_out = H - K + 1;
	const int W_out = W - K + 1;

    // Set the kernel dimensions
	int W_grid = W_out / TILE_WIDTH;
	if (W_out % TILE_WIDTH) W_grid++;
	int H_grid = H_out / TILE_WIDTH;
	if (H_out % TILE_WIDTH) H_grid++;
	int Z = H_grid * W_grid;
    dim3 gridDim(B,M,Z);
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);

    // Call the kernel
    forward_kernel<<<gridDim, blockDim>>>(y.dptr_, x.dptr_, k.dptr_, B,M,C,H,W,K);

    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

}

/*
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
{
    CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";
}
}
}

#endif
