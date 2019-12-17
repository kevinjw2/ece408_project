
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

//OPTIMIZATION: PARAMETER SWEEPING

__global__ void forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
#define BLOCK_WIDTH blockDim.x
#define BLOCK_HEIGHT blockDim.y
// helpful macros for indexing
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

const int H_out = H - K + 1;
const int W_out = W - K + 1;
int W_grid = W_out / BLOCK_WIDTH;
if (W_out % BLOCK_WIDTH) W_grid++;
int H_grid = H_out / BLOCK_HEIGHT;
if (H_out % BLOCK_HEIGHT) H_grid++;
int n, m, h, w;
n = blockIdx.z;
m = blockIdx.y;
h = (blockIdx.x / W_grid)*BLOCK_HEIGHT + threadIdx.y;
w = (blockIdx.x % W_grid)*BLOCK_WIDTH + threadIdx.x;

if(h < H_out && w < W_out) {
	float sum = 0.0;
	for (int c = 0; c < C; c++) {
		for (int p = 0; p < K; p++) {
			for (int q = 0; q < K; q++) {
					sum += x4d(n, c, h + p, w + q) * k4d(m, c, p, q);
			}
		}
	}
	y4d(n, m, h, w) = sum;
}

#undef y4d
#undef x4d
#undef k4d
#undef BLOCK_WIDTH
#undef BLOCK_HEIGHT
}

//Host code
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

    // Determine optimal block dimensions
	int block_width, block_height;
	for (int i = 16; i > 2; i--) {
		block_height = i;
		if (H_out % block_height == 0) {
			break;
		}
	}
	block_width = 32;

	int W_grid = W_out / block_width;
	int H_grid = H_out / block_height;
	if (W_out % block_width) W_grid++;
	if (H_out % block_height) H_grid++;
	int Z = H_grid * W_grid;
	dim3 blockDim(block_width, block_height, 1);
	//dim3 gridDim(B,M,Z);
    dim3 gridDim(Z,M,B);
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
