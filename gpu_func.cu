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

__global__ void sigmoid_device(const nn_real *z, nn_real *a, int array_length)
{
    int total_threads = blockDim.x * gridDim.x;
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x; idx < array_length; idx += total_threads)
        a[idx] = 1 / (1 + exp(-z[idx]));
}

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
        // exponentiate and sum over the classes
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

__global__ void kernelGEMM2D(const nn_real *A, const nn_real *B, nn_real *C, nn_real alpha, nn_real beta,
                        int M, int N, int K) 
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

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

__global__ void kernelGEMM(const nn_real *A, const nn_real *B, nn_real *C, nn_real alpha, nn_real beta,
                        int M, int N, int K) 
{
  // Block row and column
  int blockRow = blockIdx.y;
  int blockCol = blockIdx.x;

  int row = threadIdx.y;
  int col = threadIdx.x;

  int shift_C = blockRow * BLOCK_SIZE + M * blockCol * BLOCK_SIZE;
  {
    
    __shared__ nn_real C_shared[BLOCK_SIZE][BLOCK_SIZE];
    C_shared[row][col] = C[shift_C + row + col * M]; 

    nn_real res = 0;

    for (int p = 0; p < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++p) {

      __shared__ nn_real A_shared[BLOCK_SIZE][BLOCK_SIZE];
      __shared__ nn_real B_shared[BLOCK_SIZE][BLOCK_SIZE];

      int shift_A = blockRow * BLOCK_SIZE + M * p * BLOCK_SIZE;
      int shift_B = p * BLOCK_SIZE + K * blockCol * BLOCK_SIZE; 

      int idxA = shift_A + row + col * M;
      int idxB = shift_B + row + col * K;
      
      if ((blockRow * BLOCK_SIZE + row < M) && (p * BLOCK_SIZE + col < K)) 
        A_shared[row][col] = A[idxA];
      else 
        A_shared[row][col] = 0;
      
      if ((p * BLOCK_SIZE + row < K) && (blockCol * BLOCK_SIZE + col < N)) 
        B_shared[row][col] = B[idxB];
      else 
        B_shared[row][col] = 0;

      __syncthreads();

      for (int bidx = 0; bidx < BLOCK_SIZE; ++bidx)
      {
        res += A_shared[row][bidx] * B_shared[bidx][col];
      }

      __syncthreads();    
    }

    if ((blockRow * BLOCK_SIZE + row < M) && (blockCol * BLOCK_SIZE + col < N)) 
      C[shift_C + row + col * M] = alpha * res +  beta * C_shared[row][col]; 
  }
}

__global__ void ABCD_GEMMold(const nn_real *A, const nn_real *B, const nn_real *c, nn_real *D, nn_real alpha, nn_real beta,
                        int M, int N, int K) 
{
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  int col = tid / M;
  int row = tid % M;

  if (row < M && col < N)
  {
    nn_real sum = 0;  
    for (uint p = 0; p < K; ++p) 
       sum +=  A[p * M + row] * B[col * K + p];
      
    D[row + col * M] = alpha * sum + beta * c[row];
  } 
}

__global__ void ABCD_GEMM(const nn_real *A, const nn_real *B, const nn_real *c, nn_real *D, nn_real alpha, nn_real beta,
                        int M, int N, int K) 
{
  // Block row and column
  int blockRow = blockIdx.y;
  int blockCol = blockIdx.x;

  int row = threadIdx.y;
  int col = threadIdx.x;

  int shift_D = blockRow * BLOCK_SIZE + M * blockCol * BLOCK_SIZE;
  {
    __shared__ nn_real c_shared[BLOCK_SIZE];
    if (blockRow * BLOCK_SIZE + row < M)
      c_shared[row] = c[blockRow * BLOCK_SIZE + row]; 
    else
      c_shared[row] = 0;

    nn_real res = 0;

    for (int p = 0; p < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++p) {

      __shared__ nn_real A_shared[BLOCK_SIZE][BLOCK_SIZE];
      __shared__ nn_real B_shared[BLOCK_SIZE][BLOCK_SIZE];

      int shift_A = blockRow * BLOCK_SIZE + M * p * BLOCK_SIZE;
      int shift_B = p * BLOCK_SIZE + K * blockCol * BLOCK_SIZE; 

      int idxA = shift_A + row + col * M;
      int idxB = shift_B + row + col * K;
      
      if ((blockRow * BLOCK_SIZE + row < M) && (p * BLOCK_SIZE + col < K)) 
        A_shared[row][col] = A[idxA];
      else 
        A_shared[row][col] = 0;
      
      if ((p * BLOCK_SIZE + row < K) && (blockCol * BLOCK_SIZE + col < N)) 
        B_shared[row][col] = B[idxB];
      else 
        B_shared[row][col] = 0;

      __syncthreads();

      for (int bidx = 0; bidx < BLOCK_SIZE; ++bidx)
      {
        res += A_shared[row][bidx] * B_shared[bidx][col];
      }

      __syncthreads();    
    }

    if ((blockRow * BLOCK_SIZE + row < M) && (blockCol * BLOCK_SIZE + col < N)) 
      D[shift_D + row + col * M] = alpha * res +  beta * c_shared[row]; 
  }
}


__global__ void ABtCD_GEMMold(const nn_real *A, const nn_real *B, const nn_real *C, nn_real *D, nn_real alpha, nn_real beta,
                        int M, int N, int K) 
{
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  int col = tid / M;
  int row = tid % M;

  if (row < M && col < N)
  {
    nn_real sum = 0;  
    for (uint p = 0; p < K; ++p) 
      sum +=  A[p * M + row] * B[p * N + col];
       //sum +=  A[p * M + row] * B[col * K + p];
      
    D[row + col * M] = alpha * sum + beta * C[row + col * M];
  } 
}

__global__ void ABtCD_GEMM(const nn_real *A, const nn_real *B, const nn_real *C, nn_real *D, nn_real alpha, nn_real beta,
                        int M, int N, int K) 
{
  // Block row and column
  int blockRow = blockIdx.y;
  int blockCol = blockIdx.x;

  int row = threadIdx.y;
  int col = threadIdx.x;

  int shift_D = blockRow * BLOCK_SIZE + M * blockCol * BLOCK_SIZE;
  {
    __shared__ nn_real C_shared[BLOCK_SIZE][BLOCK_SIZE];
    C_shared[row][col] = C[shift_D + row + col * M]; 

    nn_real res = 0;

    for (int p = 0; p < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++p) {

      __shared__ nn_real A_shared[BLOCK_SIZE][BLOCK_SIZE];
      __shared__ nn_real B_shared[BLOCK_SIZE][BLOCK_SIZE];

      int shift_A = blockRow * BLOCK_SIZE + M * p * BLOCK_SIZE;
      int shift_B = blockCol * BLOCK_SIZE + N * p * BLOCK_SIZE;

      int idxA = shift_A + row + col * M;
      int idxB = shift_B + row + col * N;
      
      if ((blockRow * BLOCK_SIZE + row < M) && (p * BLOCK_SIZE + col < K)) 
        A_shared[row][col] = A[idxA];
      else 
        A_shared[row][col] = 0;
      
      if ((p * BLOCK_SIZE + col < K) && (blockCol * BLOCK_SIZE + row < N)) 
        B_shared[col][row] = B[idxB];
      else 
        B_shared[col][row] = 0;

      __syncthreads();

      for (int bidx = 0; bidx < BLOCK_SIZE; ++bidx)
      {
        res += A_shared[row][bidx] * B_shared[bidx][col];
      }

      __syncthreads();    
    }

    if ((blockRow * BLOCK_SIZE + row < M) && (blockCol * BLOCK_SIZE + col < N)) 
      D[shift_D + row + col * M] = alpha * res +  beta * C_shared[row][col]; 
  }
}

__global__ void AtBCD_GEMMold(const nn_real *A, const nn_real *B, const nn_real *C, nn_real *D, nn_real alpha, nn_real beta,
                        int M, int N, int K) 
{
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  int col = tid / M;
  int row = tid % M;

  if (row < M && col < N)
  {
    nn_real sum = 0;  
    for (uint p = 0; p < K; ++p) 
      sum +=  A[row * K + p] * B[col * K + p];
       //sum +=  A[p * M + row] * B[col * K + p];
      
    D[row + col * M] = alpha * sum + beta * C[row + col * M];
  } 
}

__global__ void AtBCD_GEMM(const nn_real *A, const nn_real *B, const nn_real *C, nn_real *D, nn_real alpha, nn_real beta,
                        int M, int N, int K) 
{
    // Block row and column
  int blockRow = blockIdx.y;
  int blockCol = blockIdx.x;

  int row = threadIdx.y;
  int col = threadIdx.x;

  int shift_D = blockRow * BLOCK_SIZE + M * blockCol * BLOCK_SIZE;
  {
    __shared__ nn_real C_shared[BLOCK_SIZE][BLOCK_SIZE];
    C_shared[row][col] = C[shift_D + row + col * M]; 

    nn_real res = 0;

    for (int p = 0; p < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++p) {

      __shared__ nn_real A_shared[BLOCK_SIZE][BLOCK_SIZE];
      __shared__ nn_real B_shared[BLOCK_SIZE][BLOCK_SIZE];

      //int shift_A = blockRow * BLOCK_SIZE + M * p * BLOCK_SIZE;
      int shift_A = p * BLOCK_SIZE + K * blockRow * BLOCK_SIZE;
      int shift_B = p * BLOCK_SIZE + K * blockCol * BLOCK_SIZE; 

      //int idxA = shift_A + row + col * M;
      int idxA = shift_A + row + col * K;
      int idxB = shift_B + row + col * K;
      
      if ((blockRow * BLOCK_SIZE + col < M) && (p * BLOCK_SIZE + row < K)) 
        A_shared[col][row] = A[idxA];
      else 
        A_shared[col][row] = 0;
      
      if ((p * BLOCK_SIZE + row < K) && (blockCol * BLOCK_SIZE + col < N)) 
        B_shared[row][col] = B[idxB];
      else 
        B_shared[row][col] = 0;

      __syncthreads();

      for (int bidx = 0; bidx < BLOCK_SIZE; ++bidx)
      {
        res += A_shared[row][bidx] * B_shared[bidx][col];
      }

      __syncthreads();    
    }

    if ((blockRow * BLOCK_SIZE + row < M) && (blockCol * BLOCK_SIZE + col < N)) 
      D[shift_D + row + col * M] = alpha * res +  beta * C_shared[row][col]; 
  }
}

__global__ void ABCD_elementwise(const nn_real *A, const nn_real *B, const nn_real *C, 
                                nn_real *D, nn_real alpha, int M, int N)
{
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  int col = tid / M;
  int row = tid % M;

  if (row < M && col < N)
  {
    int idx =  row + col * M;
    D[idx] = alpha * A[idx] * B[idx] * (1 - C[idx]);
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
        sum += A[row + col * M];

    B[row] = sum;
  } 
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

int myGEMM(nn_real *__restrict__ A, nn_real *__restrict__ B,
           nn_real *__restrict__ C,
           nn_real alpha, nn_real beta,
           int M, int N, int K)
{
    // Launch the kernel
    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_dim((N + block_dim.x - 1)  / block_dim.x, (M + block_dim.y - 1) / block_dim.y);

    kernelGEMM<<<grid_dim, block_dim>>>(A, B, C, alpha, beta, M, N, K);
    return 0;
}

int myABCD_GEMMold(nn_real *__restrict__ A, nn_real *__restrict__ B,
           nn_real *__restrict__ C, nn_real *D,
           nn_real alpha, nn_real beta,
           int M, int N, int K)
{
    // Launch the kernel
    size_t block_size = 256;
    size_t grid_size = (M * N + block_size - 1) / block_size;
    ABCD_GEMMold<<<grid_size, block_size>>>(A, B, C, D, alpha, beta, M, N, K);
    return 0;
}

int myABCD_GEMM(nn_real *__restrict__ A, nn_real *__restrict__ B,
           nn_real *__restrict__ C, nn_real *D,
           nn_real alpha, nn_real beta,
           int M, int N, int K)
{
    // Launch the kernel
    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_dim((N + block_dim.x - 1)  / block_dim.x, (M + block_dim.y - 1) / block_dim.y);
    ABCD_GEMM<<<grid_dim, block_dim>>>(A, B, C, D, alpha, beta, M, N, K);
    return 0;
}

int myABtCD_GEMMold(nn_real *__restrict__ A, nn_real *__restrict__ B,
           nn_real *__restrict__ C, nn_real *D,
           nn_real alpha, nn_real beta,
           int M, int N, int K)
{
    // Launch the kernel
    size_t block_size = 256;
    size_t grid_size = (M * N + block_size - 1) / block_size;
    ABtCD_GEMMold<<<grid_size, block_size>>>(A, B, C, D, alpha, beta, M, N, K);
    return 0;
}

int myABtCD_GEMM(nn_real *__restrict__ A, nn_real *__restrict__ B,
           nn_real *__restrict__ C, nn_real *D,
           nn_real alpha, nn_real beta,
           int M, int N, int K)
{
    // Launch the kernel
    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_dim((N + block_dim.x - 1)  / block_dim.x, (M + block_dim.y - 1) / block_dim.y);
    ABtCD_GEMM<<<grid_dim, block_dim>>>(A, B, C, D, alpha, beta, M, N, K);
    return 0;
}

int myAtBCD_GEMMold(nn_real *__restrict__ A, nn_real *__restrict__ B,
           nn_real *__restrict__ C, nn_real *D,
           nn_real alpha, nn_real beta,
           int M, int N, int K)
{
    // Launch the kernel
    size_t block_size = 256;
    size_t grid_size = (M * N + block_size - 1) / block_size;
    AtBCD_GEMMold<<<grid_size, block_size>>>(A, B, C, D, alpha, beta, M, N, K);
    return 0;
}

int myAtBCD_GEMM(nn_real *__restrict__ A, nn_real *__restrict__ B,
           nn_real *__restrict__ C, nn_real *D,
           nn_real alpha, nn_real beta,
           int M, int N, int K)
{
    // Launch the kernel
    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_dim((N + block_dim.x - 1)  / block_dim.x, (M + block_dim.y - 1) / block_dim.y);
    AtBCD_GEMM<<<grid_dim, block_dim>>>(A, B, C, D, alpha, beta, M, N, K);
    return 0;
}


void my_sigmoid(const nn_real *z, nn_real *a, int array_length)
{
    // Launch the kernel
    size_t block_size = 1024;
    size_t grid_size = (array_length + block_size - 1) / block_size;

    sigmoid_device<<<grid_size, block_size>>>(z, a, array_length);
}

void my_softmax(const nn_real *z, nn_real *a, int batch_size, int output_size)
{
    // Launch the kernel
    size_t block_size = 1024;
    size_t grid_size = (batch_size + block_size - 1) / block_size;

    softmax_device<<<grid_size, block_size>>>(z, a, batch_size, output_size);
}

void matrixAdd(const nn_real *A, const nn_real *B, nn_real *C, nn_real alpha, nn_real beta,
                        int M, int N)
{
    // Launch the kernel
    size_t block_size = 256;
    size_t grid_size = (M * N + block_size - 1) / block_size;
    matrixAdd_device<<<grid_size, block_size>>>(A, B, C, alpha, beta, M, N);
}

void my_reduce_A(const nn_real *A, nn_real *B, int M, int N)
{
      // Launch the kernel
    size_t block_size = 256;
    size_t grid_size = (M + block_size - 1) / block_size;
    reduce_A<<<grid_size, block_size>>>(A, B, M, N);
}

int myABCD_elementwise(nn_real *A, nn_real *B, nn_real *C, nn_real *D, nn_real alpha, int M, int N)
{
    // Launch the kernel
    size_t block_size = 256;
    size_t grid_size = (M * N + block_size - 1) / block_size;
    ABCD_elementwise<<<grid_size, block_size>>>(A, B, C, D, alpha, M, N);
    return 0;
}