#ifndef GPU_FUNC_H_
#define GPU_FUNC_H_

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include "utils/common.h"
#include "utils/gpu_util.h"

__global__ void sigmoid_device(const nn_real *z, nn_real *a, int array_length);

__global__ void softmax_device(const nn_real *z, nn_real *a, int batch_size, int output_size);

__global__ void kernelGEMMold(const nn_real *A, const nn_real *B, nn_real *C, nn_real alpha, nn_real beta,
                        int M, int N, int K); 

__global__ void kernelGEMM(const nn_real *A, const nn_real *B, nn_real *C, nn_real alpha, nn_real beta,
                        int M, int N, int K); 

__global__ void kernelGEMM2D(const nn_real *A, const nn_real *B, nn_real *C, nn_real alpha, nn_real beta,
                        int M, int N, int K);                         

__global__ void matrixAdd_device(const nn_real *A, const nn_real *B, nn_real *C, nn_real alpha, nn_real beta,
                        int M, int N); 

__global__ void ABCD_GEMM(const nn_real *A, const nn_real *B, const nn_real *c, nn_real *D, nn_real alpha, nn_real beta,
                        int M, int N, int K);

__global__ void ABtCD_GEMM(const nn_real *A, const nn_real *B, const nn_real *C, nn_real *D, nn_real alpha, nn_real beta,
                        int M, int N, int K); 

__global__ void AtBCD_GEMM(const nn_real *A, const nn_real *B, const nn_real *C, nn_real *D, nn_real alpha, nn_real beta,
                        int M, int N, int K); 

__global__ void ABCD_elementwise(const nn_real *A, const nn_real *B, const nn_real *C, nn_real *D, nn_real alpha,
                        int M, int N); 

__global__ void reduce_A(const nn_real *A, nn_real *B, int M, int N);         

struct gpu_nn
{
  nn_real *W1_d;
  nn_real *W2_d;

  nn_real *b1_d;
  nn_real *b2_d; 

  nn_real *all_X_d;
  nn_real *all_y_d;
};

struct gpu_grads
{
  nn_real *dW1_d;
  nn_real *dW2_d;

  nn_real *db1_d;
  nn_real *db2_d;

  nn_real *da1_d;
  nn_real *dz1_d;
};

struct gpu_cache
{
  nn_real *X_d;
  nn_real *y_d;
  
  nn_real *z1_d;
  nn_real *z2_d;
  nn_real *a1_d;
  nn_real *yc_d;

  nn_real *diff_y_d;
};

struct cpu_cache
{
    nn_real *dW1_partial_h;
    nn_real *dW2_partial_h;
    nn_real *db1_partial_h;
    nn_real *db2_partial_h;

    nn_real *dW1_total_h;
    nn_real *dW2_total_h;
    nn_real *db1_total_h;
    nn_real *db2_total_h;

    nn_real *X_h;
    nn_real *y_h;    
    nn_real *all_X_h;
    nn_real *all_y_h;  
};

int myGEMMold(nn_real *A, nn_real *B, nn_real *C, nn_real alpha, nn_real beta, int M, int N, int K);

int myGEMM(nn_real *A, nn_real *B, nn_real *C, nn_real alpha, nn_real beta, int M, int N, int K);

int myABCD_GEMM(nn_real *A, nn_real *B, nn_real *C, nn_real *D, nn_real alpha, nn_real beta, int M, int N, int K);

int myABtCD_GEMM(nn_real *A, nn_real *B, nn_real *C, nn_real *D, nn_real alpha, nn_real beta, int M, int N, int K);

int myAtBCD_GEMM(nn_real *A, nn_real *B, nn_real *C, nn_real *D, nn_real alpha, nn_real beta, int M, int N, int K);

int myABCD_elementwise(nn_real *A, nn_real *B, nn_real *C, nn_real *D, nn_real alpha, int M, int N);

void my_sigmoid(const nn_real *z, nn_real *a, int array_length);

void my_softmax(const nn_real *z, nn_real *a, int batch_size, int output_size);

void matrixAdd(const nn_real *A, const nn_real *B, nn_real *C, nn_real alpha, nn_real beta,
                        int M, int N); 

void my_reduce_A(const nn_real *A, nn_real *B, int M, int N);  

#endif
