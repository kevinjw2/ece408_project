
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
//__global__ void forward_kernel(float *y, const float * __restrict x, const float * __restrict k, const int B, const int M, const int C, const int H, const int W, const int K)
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

	for (int i =0; i < num_iterations; ++i){
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

		// Original indices in the input tensor.
		int X_b = b;
		int X_c = temp_row/(K*K);
		int X_p = temp_row%(K*K)/K, X_q = (temp_row%(K*K))%K;
		int X_h = column/W_out , X_w = column%W_out;
		if ((temp_row < numMatAColumns) && (column < H_out*W_out)){
			tileMatB[ty][tx] = x4d(X_b, X_c, X_h + X_p, X_w + X_q);
		}
		else{
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


// //OPTIMIZATION: LOOP UNROLLING WITH __restrict__
//
// #define TILE_WIDTH 16
//
// __global__ void forward_kernel(float * __restrict__ y, const float * __restrict__ x, const float * __restrict__ k, const int B, const int M, const int C, const int H, const int W, const int K)
// {
//
// #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
// #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
// #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
//
// const int H_out = H - K + 1;
// const int W_out = W - K + 1;
// int W_grid = W_out / TILE_WIDTH;
// if (W_out % TILE_WIDTH) W_grid++;
// int H_grid = H_out / TILE_WIDTH;
// if (H_out % TILE_WIDTH) H_grid++;
// int n, m, h, w;
// n = blockIdx.x;
// m = blockIdx.y;
// h = (blockIdx.z / W_grid)*TILE_WIDTH + threadIdx.y;
// w = (blockIdx.z % W_grid)*TILE_WIDTH + threadIdx.x;
//
// if(h < H_out && w < W_out) {
//
// 	float sum = 0.0;
//
// 	for (int c = 0; c < C; c += 3) {
// 		#pragma unroll
// 		for (int p = 0; p < K; p++) {
// 			#pragma unroll
// 			for (int q = 0; q < K; q++) {
// 				sum += x4d(n, c, h + p, w + q) * k4d(m, c, p, q);
// 				if (c+1 < C)
// 					sum += x4d(n, c+1, h + p, w + q) * k4d(m, c+1, p, q);
// 				if(c+2 < C)
// 					sum += x4d(n, c+2, h + p, w + q) * k4d(m, c+2, p, q);
// 			}
// 		}
// 	}
//
// 	y4d(n, m, h, w) = sum;
// }
//
//
// #undef y4d
// #undef x4d
// #undef k4d
// }
//
// template <>
// void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &k)
// {
//     // Extract the tensor dimensions into B,M,C,H,W,K
// 	const int B = x.shape_[0];	// Batch size
//     const int M = y.shape_[1];
//     const int C = x.shape_[1];
//     const int H = x.shape_[2];
//     const int W = x.shape_[3];
//     const int K = k.shape_[3];
// 	const int H_out = H - K + 1;
// 	const int W_out = W - K + 1;
//
//     // Set the kernel dimensions
// 	int W_grid = W_out / TILE_WIDTH;
// 	if (W_out % TILE_WIDTH) W_grid++;
// 	int H_grid = H_out / TILE_WIDTH;
// 	if (H_out % TILE_WIDTH) H_grid++;
// 	int Z = H_grid * W_grid;
//     dim3 gridDim(B,M,Z);
//     dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
//
//     // Call the kernel
//     forward_kernel<<<gridDim, blockDim>>>(y.dptr_, x.dptr_, k.dptr_, B,M,C,H,W,K);
//
//     // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
//     MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
// }


// // OPTIMIZATION 1: SHARED MEMORY CONVOLUTION (Strategy 3)
// // OPTIMIZATION 2: WEIGHT MATRIX IN CONSTANT MEMORY
//
// #define TILE_WIDTH 16
// #define CONSTANT_MEM_SIZE 60*1024
//
// __constant__ float W_Const[CONSTANT_MEM_SIZE / sizeof(float)];
//
// __global__ void forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K, int W_grid)
// {
//
// #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
// #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
// #define k4d(i3, i2, i1, i0) W_Const[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
//
// extern __shared__ float X_shared[];
// #define xs3d(i2, i1, i0) X_shared[(i2) * (TILE_WIDTH * TILE_WIDTH) + (i1) * (TILE_WIDTH) + i0]
//
// int b, m, c, h, w, p, q, h0, w0, h_base, w_base;
// const int H_out = H - K + 1;
// const int W_out = W - K + 1;
// const int MASK_RADIUS = (K-1)/2;
//
// b = blockIdx.x; //b is no. of batches
// m = blockIdx.y; //m is no. of output features
//
// h_base = (blockIdx.z / W_grid)*blockDim.y;
// h0 = threadIdx.y;
// w_base = (blockIdx.z % W_grid)*blockDim.x;
// w0 = threadIdx.x;
// h = h_base + h0;
// w = w_base + w0;
//
// //Strategy 3 - load inner input elements but not halo elements
// for(c = 0; c < C; c++){
//    xs3d(c, h0, w0) = x4d(b, c, h + MASK_RADIUS, w + MASK_RADIUS);
// }
// __syncthreads();
//
// //Compute 1 output element
// if (h < H_out && w < W_out){
// 	float sum = 0;
// 	for (c = 0; c < C; c++){
//         for (p = 0; p < K; p++){
//             for (q = 0; q < K; q++){
// 				//Global indices of current input element being processed
// 				int hx = h+p;
// 				int wx = w+q;
// 				if (hx < H && wx < W) { //This condition should be redundant, but keep it just in case
// 				// Check if input is in shared or global mem
// 	                if ((hx - h_base >= MASK_RADIUS) && (hx - h_base < MASK_RADIUS + TILE_WIDTH) &&
// 						(wx - w_base >= MASK_RADIUS) && (wx - w_base < MASK_RADIUS + TILE_WIDTH))
// 	                {
// 						//shared memory access
// 	                	sum += xs3d(c, hx - h_base - MASK_RADIUS, wx - w_base - MASK_RADIUS) * k4d(m,c,p,q);
// 	                }
// 	                else
// 	                {
// 						//global memory access
// 	                	sum += x4d(b, c, hx, wx) * k4d(m,c,p,q);
// 	                }
// 				}
//             }
//         }
//     }
//     y4d(b,m,h,w) = sum;
// }
//
// #undef y4d
// #undef x4d
// #undef k4d
// }
//
// template <>
// void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w)
// {
//     // Extract the tensor dimensions into B,M,C,H,W,K
//     const int B = x.shape_[0];
//     const int M = y.shape_[1];
//     const int C = x.shape_[1];
//     const int H = x.shape_[2];
//     const int W = x.shape_[3];
//     const int K = w.shape_[3];
//     const int H_out = H - K + 1;
//     const int W_out = W - K + 1;
//     int W_grid = W_out/TILE_WIDTH; // number of horizontal tiles per output map
//     if (W_out % TILE_WIDTH) W_grid++;
//     int H_grid = H_out/TILE_WIDTH; // number of vertical tiles per output map
//     if (H_out % TILE_WIDTH) H_grid++;
//     int Z = H_grid * W_grid;
//     dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
//     dim3 gridDim(B,M,Z);
//
//     cudaMemcpyToSymbol(W_Const, w.dptr_, M*C*K*K*sizeof(float), 0, cudaMemcpyDeviceToDevice);
//
//     size_t shmem_size = sizeof(float)*C*(TILE_WIDTH)*(TILE_WIDTH);
//
//     forward_kernel<<< gridDim, blockDim, shmem_size>>>(y.dptr_,x.dptr_,w.dptr_,B,M,C,H,W,K, W_grid);
//     // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
//     MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
// }


// //OPTIMIZATION: PARAMETER SWEEPING
//
// __global__ void forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
// {
// #define BLOCK_WIDTH blockDim.x
// #define BLOCK_HEIGHT blockDim.y
// // helpful macros for indexing
// #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
// #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
// #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
//
// const int H_out = H - K + 1;
// const int W_out = W - K + 1;
// int W_grid = W_out / BLOCK_WIDTH;
// if (W_out % BLOCK_WIDTH) W_grid++;
// int H_grid = H_out / BLOCK_HEIGHT;
// if (H_out % BLOCK_HEIGHT) H_grid++;
// int n, m, h, w;
// n = blockIdx.z;
// m = blockIdx.y;
// h = (blockIdx.x / W_grid)*BLOCK_HEIGHT + threadIdx.y;
// w = (blockIdx.x % W_grid)*BLOCK_WIDTH + threadIdx.x;
//
// if(h < H_out && w < W_out) {
// 	float sum = 0.0;
// 	for (int c = 0; c < C; c++) {
// 		for (int p = 0; p < K; p++) {
// 			for (int q = 0; q < K; q++) {
// 					sum += x4d(n, c, h + p, w + q) * k4d(m, c, p, q);
// 			}
// 		}
// 	}
// 	y4d(n, m, h, w) = sum;
// }
//
// #undef y4d
// #undef x4d
// #undef k4d
// #undef BLOCK_WIDTH
// #undef BLOCK_HEIGHT
// }
//
// template <>
// void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &k)
// {
//     // Extract the tensor dimensions into B,M,C,H,W,K
// 	const int B = x.shape_[0];	// Batch size
//     const int M = y.shape_[1];
//     const int C = x.shape_[1];
//     const int H = x.shape_[2];
//     const int W = x.shape_[3];
//     const int K = k.shape_[3];
// 	const int H_out = H - K + 1;
// 	const int W_out = W - K + 1;
//
//     // Determine optimal block dimensions
// 	int block_width, block_height;
// 	for (int i = 16; i > 2; i--) {
// 		block_height = i;
// 		if (H_out % block_height == 0) {
// 			break;
// 		}
// 	}
// 	block_width = 32;
//
// 	int W_grid = W_out / block_width;
// 	int H_grid = H_out / block_height;
// 	if (W_out % block_width) W_grid++;
// 	if (H_out % block_height) H_grid++;
// 	int Z = H_grid * W_grid;
// 	dim3 blockDim(block_width, block_height, 1);
// 	//dim3 gridDim(B,M,Z);
//     dim3 gridDim(Z,M,B);
//     // Call the kernel
//     forward_kernel<<<gridDim, blockDim>>>(y.dptr_, x.dptr_, k.dptr_, B,M,C,H,W,K);
//     // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
//     MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
// }

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
