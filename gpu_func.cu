#include "gpu_func.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <iostream>
#include "cublas_v2.h"

#define BLOCK_SIZE 32

/*
Routine to perform an in-place GEMM operation, i.e., C := alpha*A*B + beta*C
*/

int myGEMM(nn_real *__restrict__ A, nn_real *__restrict__ B,
           nn_real *__restrict__ C,
           nn_real alpha, nn_real beta,
           int M, int N, int K)
{
    return 0;
}

// kernel functions for computation intensive parts

__global__ void sigmoid_device(const nn *z, nn_real *a, int array_length)
{
    // this kernel is ensure all array elements will be processed; array_length is larger than the number of total threads
    int total_threads = blockDim.x * gridDim.x;
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x; idx < array_length; idx += total_threads)
        a[idx] = 1. / (1. + exp(-z[idx]));
}

// Kernel functions for softmax functions
/* parameters are
  *z - pointer to vector z(2) = W(2)*a(1) + b(2) 
  *a - pointer to final output, a(2) = softmax(z(2))
  batch_size - size of the batch
  output_size - number of neurons in the last layer, in this case - 10, i.e. number of digits from 0 to 9
*/ 

__global__ void softmax_device(const nn_real *z, nn_real *a, int batch_size, int output_size)
{
    int total_threads = blockDim.x * gridDim.x;

    for (int batch = threadIdx.x + blockDim.x * blockIdx.x; batch < batch_size; batch += total_threads)
    {
        nn_real sum = 0;
        for (int idx = batch * output_size; idx < (batch + 1) * output_size; idx++)
        {
            a[idx] = exp(z[idx]);
            sum += a[idx];
        }

        // do softmax
        for (int idx = batch * output_size; idx < (batch + 1) * output_size; idx++)
            a[idx] /= sum;
    }
}

/* parameters are
  *A - pointer to matrix A of size M*N
  *B - pointer to matrix B of size M*N
  *C - pointer to final output, C = alpha * A + beta * B,  of size M*N
*/
__global__ void matrixAdd_device(const nn_real *A, const nn_real *B, nn_real *C, nn_real alpha, nn_real beta,
                        int M, int N)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int col = tid / M;
    int row = tid % M;

    if (row < M && col < N)
    {
        int idx = row + col * M;  
        C[idx] = alpha * A[idx] + beta * B[idx];
    } 
}

__global__ void reduce_A(const nn_real *A, nn_real *B, int M, int N)
{
    int row = threadIdx.x + blockDim.x * blockIdx.x;
    if (row < M)
    {
        nn_real sum = 0;
        for (uint col = 0; col < N; col++)
            sum += A[row + col * M]; // column major

        B[row] = sum;
    } 
}

// naive GEMM kernel
__global__ void kernelGEMMold(const nn_real *A, const nn_real *B, nn_real *C, nn_real alpha, nn_real beta,
                        int M, int N, int K) 
{
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  int col = tid / M;
  int row = tid % M;

  if (row < M && col < N)
  {
    nn_real sum = 0;  
    for (uint p = 0; p < K; ++p) 
    {
       sum +=  A[p * M + row] * B[col * K + p];
    }

    C[row + col * M] = alpha * sum + beta * C[row + col * M];
  } 
}



/* Helper functions for neural networks */

// launcher for sigmoid kernels
void my_sigmoid(const nn_real *z, nn_real *a, int array_length)
{
    size_t block_size = 1024;
    size_t grid_size = (array_length + block_size - 1)/block_size;
    sigmoid_device<<<grid_size, block_size>>>(z, a, array_length);
}

void my_softmax(const nn_real *z, nn_real *a, int batch_size, int output_size)
{
    // launch the kernel 
    size_t block_size = 1024;
    size_t grid_size = (batch_size + block_size - 1) / block_size;
    softmax_device<<<grid_size, block_size>>>>(z, a, batch_size, output_size);
}

void matrixAdd(const nn_real *A, const nn_real *B, nn_real *C, nn_real alpha, nn_real beta, int M, int N)
{
    // launch matrix addition kernel
    size_t block_size = 256;
    size_t grid_size = (M * N + block_size - 1) / block_size;
    matrixAdd_device<<<grid_size, block_size>>>(A, B, M, N);
}

void my_reduce_A(const nn_real *A, nn_real *B, int M, int N)
{
      // Launch the kernel
    size_t block_size = 256;
    size_t grid_size = (M + block_size - 1) / block_size;
    reduce_A<<<grid_size, block_size>>>(A, B, M, N);
}

int myGEMMold(nn_real *__restrict__ A, nn_real *__restrict__ B,
           nn_real *__restrict__ C,
           nn_real alpha, nn_real beta,
           int M, int N, int K)
{
    // Launch the kernel
    size_t block_size = 256;
    size_t grid_size = (M * N + block_size - 1) / block_size;

    kernelGEMMold<<<grid_size, block_size>>>(A, B, C, alpha, beta, M, N, K);
    return 0;
}

int myABCD_GEMM(nn_real *__restrict__ A, nn_real *__restrict__ B,
           nn_real *__restrict__ C, nn_real *D,
           nn_real alpha, nn_real beta,
           int M, int N, int K)
{
    // Launch the kernel
    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_dim((N + block_dim.x - 1) / block_dim.x, (M + block_dim.y - 1) / block_dim.y);
    ABCD_GEMM<<<grid_dim, block_dim>>>(A, B, C, D, alpha, beta, M, N, K);
    return 0;
}