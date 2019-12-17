
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

// OPTIMIZATION 1: SHARED MEMORY CONVOLUTION (Strategy 3)
// OPTIMIZATION 2: WEIGHT MATRIX IN CONSTANT MEMORY

#define TILE_WIDTH 16
#define CONSTANT_MEM_SIZE 60*1024

__constant__ float W_Const[CONSTANT_MEM_SIZE / sizeof(float)];

__global__ void forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K, int W_grid)
{

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) W_Const[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

extern __shared__ float X_shared[];
#define xs3d(i2, i1, i0) X_shared[(i2) * (TILE_WIDTH * TILE_WIDTH) + (i1) * (TILE_WIDTH) + i0]


int b, m, c, h, w, p, q, h0, w0, h_base, w_base;
const int H_out = H - K + 1;
const int W_out = W - K + 1;
const int MASK_RADIUS = (K-1)/2;

b = blockIdx.x; //b is no. of batches
m = blockIdx.y; //m is no. of output features

h_base = (blockIdx.z / W_grid)*blockDim.y;
h0 = threadIdx.y;
w_base = (blockIdx.z % W_grid)*blockDim.x;
w0 = threadIdx.x;
h = h_base + h0;
w = w_base + w0;

//Strategy 3 - load inner input elements but not halo elements
for(c = 0; c < C; c++){
   xs3d(c, h0, w0) = x4d(b, c, h + MASK_RADIUS, w + MASK_RADIUS);
}
__syncthreads();

//Compute 1 output element
if (h < H_out && w < W_out){
	float sum = 0;
	for (c = 0; c < C; c++){
        for (p = 0; p < K; p++){
            for (q = 0; q < K; q++){
				//Global indices of current input element being processed
				int hx = h+p;
				int wx = w+q;
				if (hx < H && wx < W) { //This condition should be redundant, but keep it just in case
				// Check if input is in shared or global mem
	                if ((hx - h_base >= MASK_RADIUS) && (hx - h_base < MASK_RADIUS + TILE_WIDTH) &&
						(wx - w_base >= MASK_RADIUS) && (wx - w_base < MASK_RADIUS + TILE_WIDTH))
	                {
						//shared memory access
	                	sum += xs3d(c, hx - h_base - MASK_RADIUS, wx - w_base - MASK_RADIUS) * k4d(m,c,p,q);
	                }
	                else
	                {
						//global memory access
	                	sum += x4d(b, c, hx, wx) * k4d(m,c,p,q);
	                }
				}
            }
        }
    }
    y4d(b,m,h,w) = sum;
}

#undef y4d
#undef x4d
#undef k4d
}

// Host code
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w)
{
    // Extract the tensor dimensions into B,M,C,H,W,K
    const int B = x.shape_[0];
    const int M = y.shape_[1];
    const int C = x.shape_[1];
    const int H = x.shape_[2];
    const int W = x.shape_[3];
    const int K = w.shape_[3];
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    int W_grid = W_out/TILE_WIDTH; // number of horizontal tiles per output map
    if (W_out % TILE_WIDTH) W_grid++;
    int H_grid = H_out/TILE_WIDTH; // number of vertical tiles per output map
    if (H_out % TILE_WIDTH) H_grid++;
    int Z = H_grid * W_grid;
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim(B,M,Z);

    cudaMemcpyToSymbol(W_Const, w.dptr_, M*C*K*K*sizeof(float), 0, cudaMemcpyDeviceToDevice);

    size_t shmem_size = sizeof(float)*C*(TILE_WIDTH)*(TILE_WIDTH);

    forward_kernel<<< gridDim, blockDim, shmem_size>>>(y.dptr_,x.dptr_,w.dptr_,B,M,C,H,W,K, W_grid);
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
