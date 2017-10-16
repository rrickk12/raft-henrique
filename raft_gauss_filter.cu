/*
    Henrique Machado Gonçalves - 09/2017
    LNLS - CNPEM : Campinas - São Paulo - Brasil.

    This code is a implementation of a gaussian convolution filter using
    separable kernels to optimize the computer performance.


    This program was made using techniques specifically
    for the Pascal architecture of NVIDIA cards. The data acess policy
    might change in others types of architecture.

    The high-priority recommendations are as follows:
      Find ways to parallelize sequential code,
      Minimize data transfers between the host and the device,
      Adjust kernel launch configuration to maximize device utilization,
      Ensure global memory accesses are coalesced,
      Minimize redundant accesses to global memory whenever possible,
      Avoid long sequences of diverged execution by threads within the same warp.

      The Pascal Streaming Multiprocessor (SM) is in many respects similar to that of Maxwell.

      This architecture have HBM2 memories provide dedicated ECC resources,
      allowing overhead-free ECC protection, there is no need to turn off
      SECDED for performance enhancement.

      By default, GP100 caches global loads in the L1/Texture cache.
      Which acts as a coalescing buffer for memory acesses. If you are
      currently using an GP104 use -Xptxas -dlcm=ca flag to nvcc at compile time

      Is no longer necessary to turn off L1 caching
      in order to reduce wasted global memory transactions
      associated with uncoalesced accesses.
      Two new device attributes were added in CUDA Toolkit 6.0:
       -globalL1CacheSupported and localL1CacheSupported.

      The cudaDeviceEnablePeerAccess() API call remains necessary
      to enable direct transfers (over either PCIe or NVLink) between GPUs.
      The cudaDeviceCanAccessPeer() can be used to determine if peer access
      is possible between any pair of GPUs.
*/
#include <math.h>

#include "raft_filter.h"
#include "../../../../../usr/local/cuda/include/cuda_runtime.h"
#include "../../../../../usr/local/cuda/include/curand_mtgp32_kernel.h"


static int imax( int a, int b )
{
	if ( a > b )
		return a;
	else
		return b;
}

static int imin( int a, int b )
{
	if ( a > b )
		return b;
	else
		return a;
}
/////////////////////////////////////////////////////////////////////////////////////////////
// these are extremely important for performance issues
// if the READ and WRITE batches of GPU devices change, you may change the values of
// blocks dims to enhance the speed up
// these values are tuned for processing on a tesla p100 gpu.
/////////////////////////////////////////////////////////////////////////////////////////////
/// CONSTANTS FOR KERNELS 3x3x3
#define KERNEL_RADIUS_3 1
#define KERNEL_LENGTH_3 (2 * KERNEL_RADIUS_3 + 1)
// how many threads per block in x (total num threads: x*y)
#define	ROWS_BLOCKDIM_X_3 32
// how many threads per block in y
#define	ROWS_BLOCKDIM_Y_3 16
// how many pixels in x are convolved by each thread
#define	ROWS_RESULT_STEPS_3 8
// these are the border pixels (loaded to support the kernel width for processing)
// the effective border width is ROWS_HALO_STEPS_3 * ROWS_BLOCKDIM_X_3, which has to be
// larger or equal to the kernel radius to work
#define	ROWS_HALO_STEPS_3 1

#define	COLUMNS_BLOCKDIM_X_3 32
#define	COLUMNS_BLOCKDIM_Y_3 16
#define	COLUMNS_RESULT_STEPS_3 8
#define	COLUMNS_HALO_STEPS_3 1

#define	DEPTH_BLOCKDIM_X_3 64
#define	DEPTH_BLOCKDIM_Z_3 2
#define	DEPTH_RESULT_STEPS_3 8
#define	DEPTH_HALO_STEPS_3 1

/////////////////////////////////////////////////////////////////////////////////////////////
/// CONSTANTS FOR KERNELS 15x15x15
#define KERNEL_RADIUS_15 7
#define KERNEL_LENGTH_15 (2 * KERNEL_RADIUS_15 + 1)

// how many threads per block in x (total num threads: x*y)
#define	ROWS_BLOCKDIM_X_15 8
// how many threads per block in y
#define	ROWS_BLOCKDIM_Y_15 32
// how many pixels in x are convolved by each thread
#define	ROWS_RESULT_STEPS_15 8
// these are the border pixels (loaded to support the kernel width for processing)
// the effective border width is ROWS_HALO_STEPS_15 * ROWS_BLOCKDIM_X_15, which has to be
// larger or equal to the kernel radius to work
#define	ROWS_HALO_STEPS_15 1

#define	COLUMNS_BLOCKDIM_X_15 32
#define	COLUMNS_BLOCKDIM_Y_15 8
#define	COLUMNS_RESULT_STEPS_15 8
#define	COLUMNS_HALO_STEPS 1

#define	DEPTH_BLOCKDIM_X_15 32
#define	DEPTH_BLOCKDIM_Z_15 8
#define	DEPTH_RESULT_STEPS_15 8
#define	DEPTH_HALO_STEPS_15 1

/////////////////////////////////////////////////////////////////////////////////////////////
/// CONSTANTS FOR KERNELS 31x31x31
#define KERNEL_RADIUS_31 15
#define KERNEL_LENGTH_31 (2 * KERNEL_RADIUS_31 + 1)

// how many threads per block in x (total num threads: x*y)
#define	ROWS_BLOCKDIM_X_31 16
// how many threads per block in y
#define	ROWS_BLOCKDIM_Y_31 16
// how many pixels in x are convolved by each thread
#define	ROWS_RESULT_STEPS_31 8
// these are the border pixels (loaded to support the kernel width for processing)
// the effective border width is ROWS_HALO_STEPS_31 * ROWS_BLOCKDIM_X_31, which has to be
// larger or equal to the kernel radius to work
#define	ROWS_HALO_STEPS_31 1

#define	COLUMNS_BLOCKDIM_X_31 16
#define	COLUMNS_BLOCKDIM_Y_31 16
#define	COLUMNS_RESULT_STEPS_31 8
#define	COLUMNS_HALO_STEPS 1

#define	DEPTH_BLOCKDIM_X_31 16
#define	DEPTH_BLOCKDIM_Z_31 16
#define	DEPTH_RESULT_STEPS_31 8
#define	DEPTH_HALO_STEPS_31 1
/////////////////////////////////////////////////////////////////////////////////////////////


float raft_gaussian_3D_function(raft_fimage *gaussian_kernel, float sigma, int i){

    float exponent, exponentiation, func_gauss;
    exponent=(RAFT_SQUARE((raft_get_xcoord(gaussian_kernel, i)-(gaussian_kernel->xsize/2)))+RAFT_SQUARE((raft_get_ycoord(gaussian_kernel, i)-(gaussian_kernel->ysize/2)))+RAFT_SQUARE((raft_get_zcoord(gaussian_kernel, i)-(gaussian_kernel->zsize/2))) )/ (2*RAFT_SQUARE(sigma));
    exponentiation = expf(-exponent);
    func_gauss = 1/(2*RAFT_PI*RAFT_SQUARE(sigma))*exponentiation;

    return func_gauss;
}
raft_fimage *raft_3D_gaussian_kernel(int kernel_xsize, int kernel_ysize, int kernel_zsize, float sigma){

    raft_fimage *gaussian_kernel = raft_create_fimage(kernel_xsize,kernel_ysize,kernel_zsize);
    float kernel_normalization = 0;
    int i;

    //loop to fulfill the gaussian kernel matrix
    for(i=0;i<gaussian_kernel->n;i++){
        gaussian_kernel->val[i] = raft_gaussian_3D_function(gaussian_kernel, sigma, i);
        kernel_normalization += gaussian_kernel->val[i];
    }

    //loop to normalizate the gaussian_kernel result
    for(i=0; i<gaussian_kernel->n; i++){
        gaussian_kernel->val[i]/= kernel_normalization;
    }

    return gaussian_kernel;
}

__global__ void convolutionX_Kernel_3( float *d_Dst, float *d_Src, int imageW, int imageH, int imageD, float *c_Kernel)
{
    __shared__ float s_Data[ROWS_BLOCKDIM_Y_3][(ROWS_RESULT_STEPS_3 + 2 * ROWS_HALO_STEPS_3) * ROWS_BLOCKDIM_X_3];

    const int baseX = (blockIdx.x * ROWS_RESULT_STEPS_3 - ROWS_HALO_STEPS_3) * ROWS_BLOCKDIM_X_3 + threadIdx.x;
    const int baseY = blockIdx.y * ROWS_BLOCKDIM_Y_3 + threadIdx.y;
    const int baseZ = blockIdx.z;

    const int firstPixelInLine = ROWS_BLOCKDIM_X_3 * ROWS_HALO_STEPS_3 - threadIdx.x;
    const int lastPixelInLine = imageW - baseX - 1;

    // set the input and output arrays to the right offset (actually the output is not at the right offset, but this is corrected later)
    d_Src += baseZ * imageH * imageW + baseY * imageW + baseX;
    d_Dst += baseZ * imageH * imageW + baseY * imageW + baseX;

    // Load main data
    // Start copying after the ROWS_HALO_STEPS_3, only the original data that will be convolved
#pragma unroll

    for (int i = ROWS_HALO_STEPS_3; i < ROWS_HALO_STEPS_3 + ROWS_RESULT_STEPS_3; i++)
    {

    	s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X_3] = (imageW - baseX > i * ROWS_BLOCKDIM_X_3) ? d_Src[i * ROWS_BLOCKDIM_X_3] : 0;

    }
    // Load left halo
    // If the data fetched is outside of the image (note: baseX can be <0 for the first block) , use a zero-out of bounds strategy
#pragma unroll

    for (int i = 0; i < ROWS_HALO_STEPS_3; i++)
    {

    	s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X_3] = (baseX >= -i * ROWS_BLOCKDIM_X_3) ? d_Src[i * ROWS_BLOCKDIM_X_3] : 0;

    }

    //Load right halo
#pragma unroll

    for (int i = ROWS_HALO_STEPS_3 + ROWS_RESULT_STEPS_3; i < ROWS_HALO_STEPS_3 + ROWS_RESULT_STEPS_3 + ROWS_HALO_STEPS_3; i++)
    {
    	s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X_3] = (imageW - baseX > i * ROWS_BLOCKDIM_X_3) ? d_Src[i * ROWS_BLOCKDIM_X_3] : 0;
    }

    //Compute and store results
    __syncthreads();

    // this pixel is not part of the image and does not need to be convolved
    if ( baseY >= imageH )
    	return;

#pragma unroll

    for (int i = ROWS_HALO_STEPS_3; i < ROWS_HALO_STEPS_3 + ROWS_RESULT_STEPS_3; i++)
    {
        if (imageW - baseX > i * ROWS_BLOCKDIM_X_3)
        {
			float sum = 0;

	#pragma unroll

			for (int j = -KERNEL_RADIUS_3; j <= KERNEL_RADIUS_3; j++)
			{
				sum += c_Kernel[KERNEL_RADIUS_3 - j] * s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X_3 + j];
			}

			d_Dst[i * ROWS_BLOCKDIM_X_3] = sum;
		}
    }
}

__global__ void convolutionY_Kernel_3( float *d_Dst, float *d_Src, int imageW, int imageH, int imageD, float *c_Kernel)
{
    __shared__ float s_Data[COLUMNS_BLOCKDIM_X_3][(COLUMNS_RESULT_STEPS_3 + 2 * COLUMNS_HALO_STEPS_3) * COLUMNS_BLOCKDIM_Y_3 + 1];

    //Offset to the upper halo edge
    const int baseX = blockIdx.x * COLUMNS_BLOCKDIM_X_3 + threadIdx.x;
    const int baseY = (blockIdx.y * COLUMNS_RESULT_STEPS_3 - COLUMNS_HALO_STEPS_3) * COLUMNS_BLOCKDIM_Y_3 + threadIdx.y;
    const int baseZ = blockIdx.z;

    const int firstPixelInLine = (COLUMNS_BLOCKDIM_Y_3 * COLUMNS_HALO_STEPS_3 - threadIdx.y) * imageW;
    const int lastPixelInLine = (imageH - baseY - 1) * imageW;

    d_Src += baseZ * imageH * imageW + baseY * imageW + baseX;
    d_Dst += baseZ * imageH * imageW + baseY * imageW + baseX;

    //Main data
#pragma unroll

    for (int i = COLUMNS_HALO_STEPS_3; i < COLUMNS_HALO_STEPS_3 + COLUMNS_RESULT_STEPS_3; i++)
    {
    	s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y_3] = (imageH - baseY > i * COLUMNS_BLOCKDIM_Y_3) ? d_Src[i * COLUMNS_BLOCKDIM_Y_3 * imageW] : 0;
    }

    //Upper halo
#pragma unroll

    for (int i = 0; i < COLUMNS_HALO_STEPS_3; i++)
    {
    		s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y_3] = (baseY >= -i * COLUMNS_BLOCKDIM_Y_3) ? d_Src[i * COLUMNS_BLOCKDIM_Y_3 * imageW] : 0;
    }

    //Lower halo
#pragma unroll

    for (int i = COLUMNS_HALO_STEPS_3 + COLUMNS_RESULT_STEPS_3; i < COLUMNS_HALO_STEPS_3 + COLUMNS_RESULT_STEPS_3 + COLUMNS_HALO_STEPS_3; i++)
    {
    	s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y_3]= (imageH - baseY > i * COLUMNS_BLOCKDIM_Y_3) ? d_Src[i * COLUMNS_BLOCKDIM_Y_3 * imageW] : 0;
    }

    //Compute and store results
    __syncthreads();

    // this pixel is not part of the image and does not need to be convolved
    if ( baseX >= imageW )
    	return;

#pragma unroll

    for (int i = COLUMNS_HALO_STEPS_3; i < COLUMNS_HALO_STEPS_3 + COLUMNS_RESULT_STEPS_3; i++)
    {
        if (imageH - baseY > i * COLUMNS_BLOCKDIM_Y_3)
        {
			float sum = 0;

		#pragma unroll

			for (int j = -KERNEL_RADIUS_3; j <= KERNEL_RADIUS_3; j++)
			{
				sum += c_Kernel[KERNEL_RADIUS_3 - j] * s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y_3 + j];
			}

    		d_Dst[i * COLUMNS_BLOCKDIM_Y_3 * imageW] = sum;
        }
    }
}

__global__ void convolutionZ_Kernel_3( float *d_Dst, float *d_Src, int imageW, int imageH, int imageD, float *c_Kernel)
{
	// here it is [x][z], we leave out y as it has a size of 1
    __shared__ float s_Data[DEPTH_BLOCKDIM_X_3][(DEPTH_RESULT_STEPS_3 + 2 * DEPTH_HALO_STEPS_3) * DEPTH_BLOCKDIM_Z_3 + 1];

    //Offset to the upper halo edge
    const int baseX = blockIdx.x * DEPTH_BLOCKDIM_X_3 + threadIdx.x;
    const int baseY = blockIdx.y;
    const int baseZ = (blockIdx.z * DEPTH_RESULT_STEPS_3 - DEPTH_HALO_STEPS_3) * DEPTH_BLOCKDIM_Z_3 + threadIdx.z;

    const int firstPixelInLine = (DEPTH_BLOCKDIM_Z_3 * DEPTH_HALO_STEPS_3 - threadIdx.z) * imageW * imageH;
    const int lastPixelInLine = (imageD - baseZ - 1) * imageW * imageH;

    d_Src += baseZ * imageH * imageW + baseY * imageW + baseX;
    d_Dst += baseZ * imageH * imageW + baseY * imageW + baseX;

    //Main data
#pragma unroll

    for (int i = DEPTH_HALO_STEPS_3; i < DEPTH_HALO_STEPS_3 + DEPTH_RESULT_STEPS_3; i++)
    {

    	s_Data[threadIdx.x][threadIdx.z + i * DEPTH_BLOCKDIM_Z_3] = (imageD - baseZ > i * DEPTH_BLOCKDIM_Z_3) ? d_Src[i * DEPTH_BLOCKDIM_Z_3 * imageW * imageH] : 0;
    }

    //Upper halo
#pragma unroll

    for (int i = 0; i < DEPTH_HALO_STEPS_3; i++)
    {
    	 s_Data[threadIdx.x][threadIdx.z + i * DEPTH_BLOCKDIM_Z_3] = (baseZ >= -i * DEPTH_BLOCKDIM_Z_3) ? d_Src[i * DEPTH_BLOCKDIM_Z_3 * imageW * imageH] : 0;
    }

    //Lower halo
#pragma unroll

    for (int i = DEPTH_HALO_STEPS_3 + DEPTH_RESULT_STEPS_3; i < DEPTH_HALO_STEPS_3 + DEPTH_RESULT_STEPS_3 + DEPTH_HALO_STEPS_3; i++)
    {
    		s_Data[threadIdx.x][threadIdx.z + i * DEPTH_BLOCKDIM_Z_3]= (imageD - baseZ > i * DEPTH_BLOCKDIM_Z_3) ? d_Src[i * DEPTH_BLOCKDIM_Z_3 * imageW * imageH] : 0;
    }

    //Compute and store results
    __syncthreads();

    // this pixel is not part of the image and does not need to be convolved
    if ( baseX >= imageW )
    	return;

#pragma unroll

    for (int i = DEPTH_HALO_STEPS_3; i < DEPTH_HALO_STEPS_3 + DEPTH_RESULT_STEPS_3; i++)
    {
        if (imageD - baseZ > i * DEPTH_BLOCKDIM_Z_3)
        {
			float sum = 0;

	#pragma unroll

			for (int j = -KERNEL_RADIUS_3; j <= KERNEL_RADIUS_3; j++)
			{
				sum += c_Kernel[KERNEL_RADIUS_3 - j] * s_Data[threadIdx.x][threadIdx.z + i * DEPTH_BLOCKDIM_Z_3 + j];
			}

        	d_Dst[i * DEPTH_BLOCKDIM_Z_3 * imageW * imageH] = sum;
        }
    }
}

void convolutionX_3( float *d_Dst, float *d_Src, int imageW, int imageH, int imageD, float *c_Kernel, cudaStream_t *stream)
{
	int blocksX = imageW / (ROWS_RESULT_STEPS_3 * ROWS_BLOCKDIM_X_3) + imin( 1, imageW % (ROWS_RESULT_STEPS_3 * ROWS_BLOCKDIM_X_3) );
	int blocksY = imageH / ROWS_BLOCKDIM_Y_3 + imin( 1, imageH % ROWS_BLOCKDIM_Y_3 );
	int blocksZ = imageD;

    dim3 blocks(blocksX, blocksY, blocksZ);
    dim3 threads(ROWS_BLOCKDIM_X_3, ROWS_BLOCKDIM_Y_3, 1);

    convolutionX_Kernel_3<<<blocks, threads, stream>>>( d_Dst, d_Src, imageW, imageH, imageD, c_Kernel);
}

void convolutionY_3( float *d_Dst, float *d_Src, int imageW, int imageH, int imageD, float *c_Kernel, cudaStream_t *stream)
{
	int blocksX = imageW / COLUMNS_BLOCKDIM_X_3 + imin( 1, imageW % COLUMNS_BLOCKDIM_X_3 );
	int blocksY = imageH / (COLUMNS_RESULT_STEPS_3 * COLUMNS_BLOCKDIM_Y_3) + imin( 1, imageH % (COLUMNS_RESULT_STEPS_3 * COLUMNS_BLOCKDIM_Y_3) );
	int blocksZ = imageD;

    dim3 blocks(blocksX, blocksY, blocksZ);
    dim3 threads(COLUMNS_BLOCKDIM_X_3, COLUMNS_BLOCKDIM_Y_3, 1);

    convolutionY_Kernel_3<<<blocks, threads, stream>>>( d_Dst, d_Src, imageW, imageH, imageD,c_Kernel);
}

void convolutionZ_3( float *d_Dst, float *d_Src, int imageW, int imageH, int imageD, float *c_Kernel, cudaStream_t *stream)
{
	int blocksX = imageW / DEPTH_BLOCKDIM_X_3 + imin(1, imageW % DEPTH_BLOCKDIM_X_3);
	int blocksY = imageH;
	int blocksZ = imageD / (DEPTH_RESULT_STEPS_3 * DEPTH_BLOCKDIM_Z_3) + imin( 1, imageD % (DEPTH_RESULT_STEPS_3 * DEPTH_BLOCKDIM_Z_3) );

    dim3 blocks(blocksX, blocksY, blocksZ);
    dim3 threads(DEPTH_BLOCKDIM_X_3, 1, DEPTH_BLOCKDIM_Z_3);

    convolutionZ_Kernel_3<<<blocks, threads, stream>>>( d_Dst, d_Src, imageW, imageH, imageD,c_Kernel);
}
__global__ void convolutionX_Kernel_31( float *d_Dst, float *d_Src, int imageW, int imageH, int imageD, float *c_Kernel)
{
    __shared__ float s_Data[ROWS_BLOCKDIM_Y_31][(ROWS_RESULT_STEPS_31 + 2 * ROWS_HALO_STEPS_31) * ROWS_BLOCKDIM_X_31];

    const int baseX = (blockIdx.x * ROWS_RESULT_STEPS_31 - ROWS_HALO_STEPS_31) * ROWS_BLOCKDIM_X_31 + threadIdx.x;
    const int baseY = blockIdx.y * ROWS_BLOCKDIM_Y_31 + threadIdx.y;
    const int baseZ = blockIdx.z;

    const int firstPixelInLine = ROWS_BLOCKDIM_X_31 * ROWS_HALO_STEPS_31 - threadIdx.x;
    const int lastPixelInLine = imageW - baseX - 1;

    // set the input and output arrays to the right offset (actually the output is not at the right offset, but this is corrected later)
    d_Src += baseZ * imageH * imageW + baseY * imageW + baseX;
    d_Dst += baseZ * imageH * imageW + baseY * imageW + baseX;

    // Load main data
    // Start copying after the ROWS_HALO_STEPS_31, only the original data that will be convolved
#pragma unroll

    for (int i = ROWS_HALO_STEPS_31; i < ROWS_HALO_STEPS_31 + ROWS_RESULT_STEPS_31; i++)
    {

    	s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X_31] = (imageW - baseX > i * ROWS_BLOCKDIM_X_31) ? d_Src[i * ROWS_BLOCKDIM_X_31] : 0;

    }
    // Load left halo
    // If the data fetched is outside of the image (note: baseX can be <0 for the first block) , use a zero-out of bounds strategy
#pragma unroll

    for (int i = 0; i < ROWS_HALO_STEPS_31; i++)
    {

    	s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X_31] = (baseX >= -i * ROWS_BLOCKDIM_X_31) ? d_Src[i * ROWS_BLOCKDIM_X_31] : 0;

    }

    //Load right halo
#pragma unroll

    for (int i = ROWS_HALO_STEPS_31 + ROWS_RESULT_STEPS_31; i < ROWS_HALO_STEPS_31 + ROWS_RESULT_STEPS_31 + ROWS_HALO_STEPS_31; i++)
    {
    	s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X_31] = (imageW - baseX > i * ROWS_BLOCKDIM_X_31) ? d_Src[i * ROWS_BLOCKDIM_X_31] : 0;
    }

    //Compute and store results
    __syncthreads();

    // this pixel is not part of the image and does not need to be convolved
    if ( baseY >= imageH )
    	return;

#pragma unroll

    for (int i = ROWS_HALO_STEPS_31; i < ROWS_HALO_STEPS_31 + ROWS_RESULT_STEPS_31; i++)
    {
        if (imageW - baseX > i * ROWS_BLOCKDIM_X_31)
        {
			float sum = 0;

	#pragma unroll

			for (int j = -KERNEL_RADIUS_31; j <= KERNEL_RADIUS_31; j++)
			{
				sum += c_Kernel[KERNEL_RADIUS_31 - j] * s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X_31 + j];
			}

			d_Dst[i * ROWS_BLOCKDIM_X_31] = sum;
		}
    }
}

__global__ void convolutionY_Kernel_31( float *d_Dst, float *d_Src, int imageW, int imageH, int imageD, float *c_Kernel)
{
    __shared__ float s_Data[COLUMNS_BLOCKDIM_X_31][(COLUMNS_RESULT_STEPS_31 + 2 * COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y_31 + 1];

    //Offset to the upper halo edge
    const int baseX = blockIdx.x * COLUMNS_BLOCKDIM_X_31 + threadIdx.x;
    const int baseY = (blockIdx.y * COLUMNS_RESULT_STEPS_31 - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y_31 + threadIdx.y;
    const int baseZ = blockIdx.z;

    const int firstPixelInLine = (COLUMNS_BLOCKDIM_Y_31 * COLUMNS_HALO_STEPS - threadIdx.y) * imageW;
    const int lastPixelInLine = (imageH - baseY - 1) * imageW;

    d_Src += baseZ * imageH * imageW + baseY * imageW + baseX;
    d_Dst += baseZ * imageH * imageW + baseY * imageW + baseX;

    //Main data
#pragma unroll

    for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS_31; i++)
    {
    	s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y_31] = (imageH - baseY > i * COLUMNS_BLOCKDIM_Y_31) ? d_Src[i * COLUMNS_BLOCKDIM_Y_31 * imageW] : 0;
    }

    //Upper halo
#pragma unroll

    for (int i = 0; i < COLUMNS_HALO_STEPS; i++)
    {
    		s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y_31] = (baseY >= -i * COLUMNS_BLOCKDIM_Y_31) ? d_Src[i * COLUMNS_BLOCKDIM_Y_31 * imageW] : 0;
    }

    //Lower halo
#pragma unroll

    for (int i = COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS_31; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS_31 + COLUMNS_HALO_STEPS; i++)
    {
    	s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y_31]= (imageH - baseY > i * COLUMNS_BLOCKDIM_Y_31) ? d_Src[i * COLUMNS_BLOCKDIM_Y_31 * imageW] : 0;
    }

    //Compute and store results
    __syncthreads();

    // this pixel is not part of the image and does not need to be convolved
    if ( baseX >= imageW )
    	return;

#pragma unroll

    for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS_31; i++)
    {
        if (imageH - baseY > i * COLUMNS_BLOCKDIM_Y_31)
        {
			float sum = 0;

		#pragma unroll

			for (int j = -KERNEL_RADIUS_31; j <= KERNEL_RADIUS_31; j++)
			{
				sum += c_Kernel[KERNEL_RADIUS_31 - j] * s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y_31 + j];
			}

    		d_Dst[i * COLUMNS_BLOCKDIM_Y_31 * imageW] = sum;
        }
    }
}

__global__ void convolutionZ_Kernel_31( float *d_Dst, float *d_Src, int imageW, int imageH, int imageD, float *c_Kernel)
{
	// here it is [x][z], we leave out y as it has a size of 1
    __shared__ float s_Data[DEPTH_BLOCKDIM_X_31][(DEPTH_RESULT_STEPS_31 + 2 * DEPTH_HALO_STEPS_31) * DEPTH_BLOCKDIM_Z_31 + 1];

    //Offset to the upper halo edge
    const int baseX = blockIdx.x * DEPTH_BLOCKDIM_X_31 + threadIdx.x;
    const int baseY = blockIdx.y;
    const int baseZ = (blockIdx.z * DEPTH_RESULT_STEPS_31 - DEPTH_HALO_STEPS_31) * DEPTH_BLOCKDIM_Z_31 + threadIdx.z;

    const int firstPixelInLine = (DEPTH_BLOCKDIM_Z_31 * DEPTH_HALO_STEPS_31 - threadIdx.z) * imageW * imageH;
    const int lastPixelInLine = (imageD - baseZ - 1) * imageW * imageH;

    d_Src += baseZ * imageH * imageW + baseY * imageW + baseX;
    d_Dst += baseZ * imageH * imageW + baseY * imageW + baseX;

    //Main data
#pragma unroll

    for (int i = DEPTH_HALO_STEPS_31; i < DEPTH_HALO_STEPS_31 + DEPTH_RESULT_STEPS_31; i++)
    {

    	s_Data[threadIdx.x][threadIdx.z + i * DEPTH_BLOCKDIM_Z_31] = (imageD - baseZ > i * DEPTH_BLOCKDIM_Z_31) ? d_Src[i * DEPTH_BLOCKDIM_Z_31 * imageW * imageH] : 0;
    }

    //Upper halo
#pragma unroll

    for (int i = 0; i < DEPTH_HALO_STEPS_31; i++)
    {
    	 s_Data[threadIdx.x][threadIdx.z + i * DEPTH_BLOCKDIM_Z_31] = (baseZ >= -i * DEPTH_BLOCKDIM_Z_31) ? d_Src[i * DEPTH_BLOCKDIM_Z_31 * imageW * imageH] : 0;
    }

    //Lower halo
#pragma unroll

    for (int i = DEPTH_HALO_STEPS_31 + DEPTH_RESULT_STEPS_31; i < DEPTH_HALO_STEPS_31 + DEPTH_RESULT_STEPS_31 + DEPTH_HALO_STEPS_31; i++)
    {
    		s_Data[threadIdx.x][threadIdx.z + i * DEPTH_BLOCKDIM_Z_31]= (imageD - baseZ > i * DEPTH_BLOCKDIM_Z_31) ? d_Src[i * DEPTH_BLOCKDIM_Z_31 * imageW * imageH] : 0;
    }

    //Compute and store results
    __syncthreads();

    // this pixel is not part of the image and does not need to be convolved
    if ( baseX >= imageW )
    	return;

#pragma unroll

    for (int i = DEPTH_HALO_STEPS_31; i < DEPTH_HALO_STEPS_31 + DEPTH_RESULT_STEPS_31; i++)
    {
        if (imageD - baseZ > i * DEPTH_BLOCKDIM_Z_31)
        {
			float sum = 0;

	#pragma unroll

			for (int j = -KERNEL_RADIUS_31; j <= KERNEL_RADIUS_31; j++)
			{
				sum += c_Kernel[KERNEL_RADIUS_31 - j] * s_Data[threadIdx.x][threadIdx.z + i * DEPTH_BLOCKDIM_Z_31 + j];
			}

        	d_Dst[i * DEPTH_BLOCKDIM_Z_31 * imageW * imageH] = sum;
        }
    }
}

void convolutionX_31( float *d_Dst, float *d_Src, int imageW, int imageH, int imageD, float *c_Kernel, cudaStream_t *stream)
{
	int blocksX = imageW / (ROWS_RESULT_STEPS_31 * ROWS_BLOCKDIM_X_31) + imin( 1, imageW % (ROWS_RESULT_STEPS_31 * ROWS_BLOCKDIM_X_31) );
	int blocksY = imageH / ROWS_BLOCKDIM_Y_31 + imin( 1, imageH % ROWS_BLOCKDIM_Y_31 );
	int blocksZ = imageD;

    dim3 blocks(blocksX, blocksY, blocksZ);
    dim3 threads(ROWS_BLOCKDIM_X_31, ROWS_BLOCKDIM_Y_31, 1);

    convolutionX_Kernel_31<<<blocks, threads, stream>>>( d_Dst, d_Src, imageW, imageH, imageD, c_Kernel);
}

void convolutionY_31( float *d_Dst, float *d_Src, int imageW, int imageH, int imageD, float *c_Kernel,cudaStream_t *stream)
{
	int blocksX = imageW / COLUMNS_BLOCKDIM_X_31 + imin( 1, imageW % COLUMNS_BLOCKDIM_X_31 );
	int blocksY = imageH / (COLUMNS_RESULT_STEPS_31 * COLUMNS_BLOCKDIM_Y_31) + imin( 1, imageH % (COLUMNS_RESULT_STEPS_31 * COLUMNS_BLOCKDIM_Y_31) );
	int blocksZ = imageD;

    dim3 blocks(blocksX, blocksY, blocksZ);
    dim3 threads(COLUMNS_BLOCKDIM_X_31, COLUMNS_BLOCKDIM_Y_31, 1);

    convolutionY_Kernel_31<<<blocks, threads, stream>>>( d_Dst, d_Src, imageW, imageH, imageD,c_Kernel);
}

void convolutionZ_31( float *d_Dst, float *d_Src, int imageW, int imageH, int imageD, float *c_Kernel,cudaStream_t *stream)
{
	int blocksX = imageW / DEPTH_BLOCKDIM_X_31 + imin(1, imageW % DEPTH_BLOCKDIM_X_31);
	int blocksY = imageH;
	int blocksZ = imageD / (DEPTH_RESULT_STEPS_31 * DEPTH_BLOCKDIM_Z_31) + imin( 1, imageD % (DEPTH_RESULT_STEPS_31 * DEPTH_BLOCKDIM_Z_31) );

    dim3 blocks(blocksX, blocksY, blocksZ);
    dim3 threads(DEPTH_BLOCKDIM_X_31, 1, DEPTH_BLOCKDIM_Z_31);

    convolutionZ_Kernel_31<<<blocks, threads ,stream>>>( d_Dst, d_Src, imageW, imageH, imageD,c_Kernel);
}

__global__ void convolutionX_Kernel_15( float *d_Dst, float *d_Src, int imageW, int imageH, int imageD, float *c_Kernel)
{
    __shared__ float s_Data[ROWS_BLOCKDIM_Y_15][(ROWS_RESULT_STEPS_15 + 2 * ROWS_HALO_STEPS_15) * ROWS_BLOCKDIM_X_15];

    const int baseX = (blockIdx.x * ROWS_RESULT_STEPS_15 - ROWS_HALO_STEPS_15) * ROWS_BLOCKDIM_X_15 + threadIdx.x;
    const int baseY = blockIdx.y * ROWS_BLOCKDIM_Y_15 + threadIdx.y;
    const int baseZ = blockIdx.z;

    const int firstPixelInLine = ROWS_BLOCKDIM_X_15 * ROWS_HALO_STEPS_15 - threadIdx.x;
    const int lastPixelInLine = imageW - baseX - 1;

    // set the input and output arrays to the right offset (actually the output is not at the right offset, but this is corrected later)
    d_Src += baseZ * imageH * imageW + baseY * imageW + baseX;
    d_Dst += baseZ * imageH * imageW + baseY * imageW + baseX;

    // Load main data
    // Start copying after the ROWS_HALO_STEPS_15, only the original data that will be convolved
#pragma unroll

    for (int i = ROWS_HALO_STEPS_15; i < ROWS_HALO_STEPS_15 + ROWS_RESULT_STEPS_15; i++)
    {

    	s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X_15] = (imageW - baseX > i * ROWS_BLOCKDIM_X_15) ? d_Src[i * ROWS_BLOCKDIM_X_15] : 0;

    }
    // Load left halo
    // If the data fetched is outside of the image (note: baseX can be <0 for the first block) , use a zero-out of bounds strategy
#pragma unroll

    for (int i = 0; i < ROWS_HALO_STEPS_15; i++)
    {

    	s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X_15] = (baseX >= -i * ROWS_BLOCKDIM_X_15) ? d_Src[i * ROWS_BLOCKDIM_X_15] : 0;

    }

    //Load right halo
#pragma unroll

    for (int i = ROWS_HALO_STEPS_15 + ROWS_RESULT_STEPS_15; i < ROWS_HALO_STEPS_15 + ROWS_RESULT_STEPS_15 + ROWS_HALO_STEPS_15; i++)
    {
    	s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X_15] = (imageW - baseX > i * ROWS_BLOCKDIM_X_15) ? d_Src[i * ROWS_BLOCKDIM_X_15] : 0;
    }

    //Compute and store results
    __syncthreads();

    // this pixel is not part of the image and does not need to be convolved
    if ( baseY >= imageH )
    	return;

#pragma unroll

    for (int i = ROWS_HALO_STEPS_15; i < ROWS_HALO_STEPS_15 + ROWS_RESULT_STEPS_15; i++)
    {
        if (imageW - baseX > i * ROWS_BLOCKDIM_X_15)
        {
			float sum = 0;

	#pragma unroll

			for (int j = -KERNEL_RADIUS_15; j <= KERNEL_RADIUS_15; j++)
			{
				sum += c_Kernel[KERNEL_RADIUS_15 - j] * s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X_15 + j];
			}

			d_Dst[i * ROWS_BLOCKDIM_X_15] = sum;
		}
    }
}

__global__ void convolutionY_Kernel_15( float *d_Dst, float *d_Src, int imageW, int imageH, int imageD, float *c_Kernel)
{
    __shared__ float s_Data[COLUMNS_BLOCKDIM_X_15][(COLUMNS_RESULT_STEPS_15 + 2 * COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y_15 + 1];

    //Offset to the upper halo edge
    const int baseX = blockIdx.x * COLUMNS_BLOCKDIM_X_15 + threadIdx.x;
    const int baseY = (blockIdx.y * COLUMNS_RESULT_STEPS_15 - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y_15 + threadIdx.y;
    const int baseZ = blockIdx.z;

    const int firstPixelInLine = (COLUMNS_BLOCKDIM_Y_15 * COLUMNS_HALO_STEPS - threadIdx.y) * imageW;
    const int lastPixelInLine = (imageH - baseY - 1) * imageW;

    d_Src += baseZ * imageH * imageW + baseY * imageW + baseX;
    d_Dst += baseZ * imageH * imageW + baseY * imageW + baseX;

    //Main data
#pragma unroll

    for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS_15; i++)
    {
    	s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y_15] = (imageH - baseY > i * COLUMNS_BLOCKDIM_Y_15) ? d_Src[i * COLUMNS_BLOCKDIM_Y_15 * imageW] : 0;
    }

    //Upper halo
#pragma unroll

    for (int i = 0; i < COLUMNS_HALO_STEPS; i++)
    {
    		s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y_15] = (baseY >= -i * COLUMNS_BLOCKDIM_Y_15) ? d_Src[i * COLUMNS_BLOCKDIM_Y_15 * imageW] : 0;
    }

    //Lower halo
#pragma unroll

    for (int i = COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS_15; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS_15 + COLUMNS_HALO_STEPS; i++)
    {
    	s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y_15]= (imageH - baseY > i * COLUMNS_BLOCKDIM_Y_15) ? d_Src[i * COLUMNS_BLOCKDIM_Y_15 * imageW] : 0;
    }

    //Compute and store results
    __syncthreads();

    // this pixel is not part of the image and does not need to be convolved
    if ( baseX >= imageW )
    	return;

#pragma unroll

    for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS_15; i++)
    {
        if (imageH - baseY > i * COLUMNS_BLOCKDIM_Y_15)
        {
			float sum = 0;

		#pragma unroll

			for (int j = -KERNEL_RADIUS_15; j <= KERNEL_RADIUS_15; j++)
			{
				sum += c_Kernel[KERNEL_RADIUS_15 - j] * s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y_15 + j];
			}

    		d_Dst[i * COLUMNS_BLOCKDIM_Y_15 * imageW] = sum;
        }
    }
}

__global__ void convolutionZ_Kernel_15( float *d_Dst, float *d_Src, int imageW, int imageH, int imageD, float *c_Kernel)
{
	// here it is [x][z], we leave out y as it has a size of 1
    __shared__ float s_Data[DEPTH_BLOCKDIM_X_15][(DEPTH_RESULT_STEPS_15 + 2 * DEPTH_HALO_STEPS_15) * DEPTH_BLOCKDIM_Z_15 + 1];

    //Offset to the upper halo edge
    const int baseX = blockIdx.x * DEPTH_BLOCKDIM_X_15 + threadIdx.x;
    const int baseY = blockIdx.y;
    const int baseZ = (blockIdx.z * DEPTH_RESULT_STEPS_15 - DEPTH_HALO_STEPS_15) * DEPTH_BLOCKDIM_Z_15 + threadIdx.z;

    const int firstPixelInLine = (DEPTH_BLOCKDIM_Z_15 * DEPTH_HALO_STEPS_15 - threadIdx.z) * imageW * imageH;
    const int lastPixelInLine = (imageD - baseZ - 1) * imageW * imageH;

    d_Src += baseZ * imageH * imageW + baseY * imageW + baseX;
    d_Dst += baseZ * imageH * imageW + baseY * imageW + baseX;

    //Main data
#pragma unroll

    for (int i = DEPTH_HALO_STEPS_15; i < DEPTH_HALO_STEPS_15 + DEPTH_RESULT_STEPS_15; i++)
    {

    	s_Data[threadIdx.x][threadIdx.z + i * DEPTH_BLOCKDIM_Z_15] = (imageD - baseZ > i * DEPTH_BLOCKDIM_Z_15) ? d_Src[i * DEPTH_BLOCKDIM_Z_15 * imageW * imageH] : 0;
    }

    //Upper halo
#pragma unroll

    for (int i = 0; i < DEPTH_HALO_STEPS_15; i++)
    {
    	 s_Data[threadIdx.x][threadIdx.z + i * DEPTH_BLOCKDIM_Z_15] = (baseZ >= -i * DEPTH_BLOCKDIM_Z_15) ? d_Src[i * DEPTH_BLOCKDIM_Z_15 * imageW * imageH] : 0;
    }

    //Lower halo
#pragma unroll

    for (int i = DEPTH_HALO_STEPS_15 + DEPTH_RESULT_STEPS_15; i < DEPTH_HALO_STEPS_15 + DEPTH_RESULT_STEPS_15 + DEPTH_HALO_STEPS_15; i++)
    {
    		s_Data[threadIdx.x][threadIdx.z + i * DEPTH_BLOCKDIM_Z_15]= (imageD - baseZ > i * DEPTH_BLOCKDIM_Z_15) ? d_Src[i * DEPTH_BLOCKDIM_Z_15 * imageW * imageH] : 0;
    }

    //Compute and store results
    __syncthreads();

    // this pixel is not part of the image and does not need to be convolved
    if ( baseX >= imageW )
    	return;

#pragma unroll

    for (int i = DEPTH_HALO_STEPS_15; i < DEPTH_HALO_STEPS_15 + DEPTH_RESULT_STEPS_15; i++)
    {
        if (imageD - baseZ > i * DEPTH_BLOCKDIM_Z_15)
        {
			float sum = 0;

	#pragma unroll

			for (int j = -KERNEL_RADIUS_15; j <= KERNEL_RADIUS_15; j++)
			{
				sum += c_Kernel[KERNEL_RADIUS_15 - j] * s_Data[threadIdx.x][threadIdx.z + i * DEPTH_BLOCKDIM_Z_15 + j];
			}

        	d_Dst[i * DEPTH_BLOCKDIM_Z_15 * imageW * imageH] = sum;
        }
    }
}

void convolutionX_15( float *d_Dst, float *d_Src, int imageW, int imageH, int imageD, float *c_Kernel,, cudaStream_t *stream)
{
	int blocksX = imageW / (ROWS_RESULT_STEPS_15 * ROWS_BLOCKDIM_X_15) + imin( 1, imageW % (ROWS_RESULT_STEPS_15 * ROWS_BLOCKDIM_X_15) );
	int blocksY = imageH / ROWS_BLOCKDIM_Y_15 + imin( 1, imageH % ROWS_BLOCKDIM_Y_15 );
	int blocksZ = imageD;

    dim3 blocks(blocksX, blocksY, blocksZ);
    dim3 threads(ROWS_BLOCKDIM_X_15, ROWS_BLOCKDIM_Y_15, 1);

    convolutionX_Kernel_15<<<blocks, threads, stream>>>( d_Dst, d_Src, imageW, imageH, imageD, c_Kernel);
}

void convolutionY_15( float *d_Dst, float *d_Src, int imageW, int imageH, int imageD, float *c_Kernel, cudaStream_t *stream)
{
	int blocksX = imageW / COLUMNS_BLOCKDIM_X_15 + imin( 1, imageW % COLUMNS_BLOCKDIM_X_15 );
	int blocksY = imageH / (COLUMNS_RESULT_STEPS_15 * COLUMNS_BLOCKDIM_Y_15) + imin( 1, imageH % (COLUMNS_RESULT_STEPS_15 * COLUMNS_BLOCKDIM_Y_15) );
	int blocksZ = imageD;

    dim3 blocks(blocksX, blocksY, blocksZ);
    dim3 threads(COLUMNS_BLOCKDIM_X_15, COLUMNS_BLOCKDIM_Y_15, 1);

    convolutionY_Kernel_15<<<blocks, threads,stream>>>( d_Dst, d_Src, imageW, imageH, imageD,c_Kernel);
}

void convolutionZ_15( float *d_Dst, float *d_Src, int imageW, int imageH, int imageD, float *c_Kernel, cudaStream_t *stream)
{
	int blocksX = imageW / DEPTH_BLOCKDIM_X_15 + imin(1, imageW % DEPTH_BLOCKDIM_X_15);
	int blocksY = imageH;
	int blocksZ = imageD / (DEPTH_RESULT_STEPS_15 * DEPTH_BLOCKDIM_Z_15) + imin( 1, imageD % (DEPTH_RESULT_STEPS_15 * DEPTH_BLOCKDIM_Z_15) );

    dim3 blocks(blocksX, blocksY, blocksZ);
    dim3 threads(DEPTH_BLOCKDIM_X_15, 1, DEPTH_BLOCKDIM_Z_15);

    convolutionZ_Kernel_15<<<blocks, threads,stream>>>( d_Dst, d_Src, imageW, imageH, imageD,c_Kernel);
}
int main(float sigma)
{
    //time variables
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float milliseconds = 0;
	//sigma define which kernel we are going to apply in the image
	int kernel_type = 0;
  float *d_gaussian_kernelx,*d_gaussian_kernely,*d_gaussian_kernelz, *d_image, *d_out_image;
	raft_fimage *gaussian_kernelx;
	raft_fimage *gaussian_kernely;
	raft_fimage *gaussian_kernelz;
	if(sigma * 4 <= 3.0 )
	{
		kernel_type = 3;
	}
	else if(sigma *4 <= 15.0)
	{
		kernel_type = 15;
	}
	else
	{
		kernel_type = 31;
	}

  raft_fimage *image = raft_fread_raw_slices("slice_", 1000, 1049, 2048, 2048);
	if (kernel_type == 3)
	{
		gaussian_kernelx = raft_3D_gaussian_kernel(KERNEL_LENGTH_3,1,1,sigma);
		gaussian_kernely = raft_3D_gaussian_kernel(1,KERNEL_LENGTH_3,1,sigma);
		gaussian_kernelz = raft_3D_gaussian_kernel(1,1,KERNEL_LENGTH_3,sigma);
	}
	else if (kernel_type == 15)
	{
		gaussian_kernelx = raft_3D_gaussian_kernel(KERNEL_LENGTH_15,1,1,sigma);
		gaussian_kernely = raft_3D_gaussian_kernel(1,KERNEL_LENGTH_15,1,sigma);
		gaussian_kernelz = raft_3D_gaussian_kernel(1,1,KERNEL_LENGTH_15,sigma);
	}
	else if (kernel_type == 31)
	{
		gaussian_kernelx = raft_3D_gaussian_kernel(KERNEL_LENGTH_31,1,1,sigma);
		gaussian_kernely = raft_3D_gaussian_kernel(1,KERNEL_LENGTH_31,1,sigma);
		gaussian_kernelz = raft_3D_gaussian_kernel(1,1,KERNEL_LENGTH_31,sigma);
	}

  raft_fimage *out_image = raft_create_fimage(image->xsize, image->ysize, image->zsize);

  d_gaussian_kernelx = raft_cuda_alloc_float_array(gaussian_kernelx->n, false, false);
  d_gaussian_kernely = raft_cuda_alloc_float_array(gaussian_kernely->n, false, false);
  d_gaussian_kernelz = raft_cuda_alloc_float_array(gaussian_kernelz->n, false, false);
  d_image = raft_cuda_alloc_float_array(image->n, false, false);
  d_out_image = raft_cuda_alloc_float_array(out_image->n, false, false );

  //create cuda Streams
  cudaStream_t stream[5];
  cudaStreamCreateWithFlags(&stream[0],cudaStreamNonBlocking);
  cudaStreamCreateWithFlags(&stream[1],cudaStreamNonBlocking);
  cudaStreamCreateWithFlags(&stream[2],cudaStreamNonBlocking);
  cudaStreamCreateWithFlags(&stream[3],cudaStreamNonBlocking);
  cudaStreamCreateWithFlags(&stream[4],cudaStreamNonBlocking);
  cudaEventRecord(start);

  cudaMemcpyAsync(d_image, image->val, image->n*sizeof(float), cudaMemcpyHostToDevice,stream[0]);
  cudaMemcpyAsync(d_out_image, out_image->val, out_image->n*sizeof(float), cudaMemcpyHostToDevice,stream[1]);
  cudaMemcpyAsync(d_gaussian_kernelx, gaussian_kernelx->val, gaussian_kernelx->n*sizeof(float), cudaMemcpyHostToDevice,stream[2]);
  cudaMemcpyAsync(d_gaussian_kernely, gaussian_kernely->val, gaussian_kernely->n*sizeof(float), cudaMemcpyHostToDevice,stream[3]);
  cudaMemcpyAsync(d_gaussian_kernelz, gaussian_kernelz->val, gaussian_kernelz->n*sizeof(float), cudaMemcpyHostToDevice,stream[4]);

  cudaDeviceSynchronize();

	if(kernel_type == 3)
	{
		convolutionX_3( d_out_image, d_image, image->xsize,image->ysize,image->zsize,d_gaussian_kernelx,&stream[0]);
		convolutionY_3( d_out_image, d_image, image->xsize,image->ysize,image->zsize,d_gaussian_kernely,&stream[1]);
		convolutionZ_3( d_out_image, d_image, image->xsize,image->ysize,image->zsize,d_gaussian_kernelz,&stream[2]);
	}
	if(kernel_type == 15)
	{
		convolutionX_15( d_out_image, d_image, image->xsize,image->ysize,image->zsize,d_gaussian_kernelx,&stream[0]);
		convolutionY_15( d_out_image, d_image, image->xsize,image->ysize,image->zsize,d_gaussian_kernely,&stream[1]);
		convolutionZ_15( d_out_image, d_image, image->xsize,image->ysize,image->zsize,d_gaussian_kernelz,&stream[2]);
	}
	if(kernel_type == 31)
	{
		convolutionX_31( d_out_image, d_image, image->xsize,image->ysize,image->zsize,d_gaussian_kernelx,stream[0]);
		convolutionY_31( d_out_image, d_image, image->xsize,image->ysize,image->zsize,d_gaussian_kernely,stream[1]);
		convolutionZ_31( d_out_image, d_image, image->xsize,image->ysize,image->zsize,d_gaussian_kernelz,stream[2]);
	}

  cudaDeviceSynchronize();
  cudaStreamDestroy(stream[0]);
  cudaStreamDestroy(stream[1]);
  cudaStreamDestroy(stream[2]);
  cudaStreamDestroy(stream[3]);
  cudaStreamDestroy(stream[4]);
  raft_destroy_fimage(&gaussian_kernelz);
  raft_destroy_fimage(&gaussian_kernely);
  raft_destroy_fimage(&gaussian_kernelx);
  cudaEventRecord(stop);

  cudaMemcpy(out_image->val, d_out_image, out_image->n*sizeof(float), cudaMemcpyDeviceToHost);

  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);

  printf("The time of executation on GPU was: %.2f \n",milliseconds);

  raft_fwrite_raw_image(out_image, "out_optimized3_slice.b");

  raft_destroy_fimage(&image);
  //raft_destroy_fimage(&gaussian_kernel);
  raft_destroy_fimage(&out_image);

  cudaFree(d_gaussian_kernelx);
  cudaFree(d_gaussian_kernely);
  cudaFree(d_gaussian_kernelz);
  cudaFree(d_image);
  cudaFree(d_out_image);

  return 0;
}
