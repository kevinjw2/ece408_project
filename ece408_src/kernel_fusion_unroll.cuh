
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

//OPTIMIZATION: KERNEL FUSION WITH UNROLLING/MATRIX MULTIPLICATION

#define TILE_WIDTH 16

__global__ void forward_kernel(float *y, const float *  x, const float * k, const int B, const int M, const int C, const int H, const int W, const int K)
{
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

__shared__ float tileMatA[TILE_WIDTH][TILE_WIDTH];
__shared__ float tileMatB[TILE_WIDTH][TILE_WIDTH];
const int H_out = H - K + 1;
const int W_out = W - K + 1;

int b = blockIdx.z; //batches
int tx = threadIdx.x;
int ty = threadIdx.y;
int row = blockIdx.y * TILE_WIDTH + ty;
int column = blockIdx.x * TILE_WIDTH + tx;
int numMatAColumns = C*K*K;
float acc = 0.0;
int num_iterations = ceil(1.0*numMatAColumns/TILE_WIDTH);
	//#pragma unroll_completely
	for(int i =0; i < num_iterations; ++i){
		int temp_col = i*TILE_WIDTH + tx;
	        int temp_row = i*TILE_WIDTH + ty;
		tileMatA[ty][tx] = 0;
		tileMatB[ty][tx] = 0;
		// Original indices in the filter tensor.
		int W_m = row;
		int W_c = temp_col/(K*K);
		int W_h = (temp_col%(K*K))/K;
	        int W_w = (temp_col%(K*K))%K;
		if ((temp_col < numMatAColumns) && (row < M)){
			tileMatA[ty][tx] = k4d(W_m, W_c, W_h, W_w);
		}
		else {
			tileMatA[ty][tx] = 0;
        }
		//__syncthreads();
		// Original indices in the input tensor.
		int X_b = b;
		int X_c = temp_row/(K*K);
		int X_p = temp_row%(K*K)/K, X_q = (temp_row%(K*K))%K;
		int X_h = column/W_out , X_w = column%W_out;
		if ((temp_row < numMatAColumns) && (column < H_out*W_out)){
			tileMatB[ty][tx] = x4d(X_b, X_c, X_h + X_p, X_w + X_q);
		}
		else {
			tileMatB[ty][tx] = 0;
		}
		__syncthreads();

		for (int q = 0; q < TILE_WIDTH; q++){
			acc += tileMatA[ty][q] * tileMatB[q][tx];
		}
		__syncthreads();
	}

//Original indices in the output tensor.
int Y_h = column / W_out, Y_w = column % W_out;
if (row < M && column < W_out*H_out){
y4d(b, row, Y_h, Y_w) = acc;
}
#undef y4d
#undef x4d
#undef k4d
}

template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &k)
{


    const int B = x.shape_[0];      // Batch size
    const int M = y.shape_[1];
    const int C = x.shape_[1];
    const int H = x.shape_[2];
    const int W = x.shape_[3];
    const int K = k.shape_[3];
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    // Set the kernel dimensions
    dim3 gridDim(ceil(1.0*W_out*H_out/TILE_WIDTH),ceil(1.0*M/TILE_WIDTH),B);
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
