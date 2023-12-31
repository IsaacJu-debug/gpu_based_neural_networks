#ifndef GPU_FUNC_H_
#define GPU_FUNC_H_

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include "utils/common.h"
#include "utils/gpu_util.h"

int myGEMM(nn_real *A, nn_real *B, nn_real *C, nn_real alpha, nn_real beta,
        int M, int N, int K);

// TODO
// Add additional function declarations

__global__ void sigmoid_device(const nn_real * z, nn_real * a, int array_length);

__global__ void softmax_device(const nn_real * z, nn_real * a, int batch_size, int output_size);

__global__ void kernelGEMM(const nn_real *A, const nn_real *B, nn_real *C, nn_real alpha, nn_real beta, int M, int N, int K);



// struct for hosting nn parameters, gradients, and other variables
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
    nn_real *da2_d;
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



#endif
