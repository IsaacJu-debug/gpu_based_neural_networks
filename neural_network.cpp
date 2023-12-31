#include "utils/neural_network.h"

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include <armadillo>
#include <iomanip>

#include "cublas_v2.h"
#include "gpu_func.h"
#include "mpi.h"

#define MPI_SAFE_CALL(call)                                                  \
  do                                                                         \
  {                                                                          \
    int err = call;                                                          \
    if (err != MPI_SUCCESS)                                                  \
    {                                                                        \
      fprintf(stderr, "MPI error %d in file '%s' at line %i", err, __FILE__, \
              __LINE__);                                                     \
      exit(1);                                                               \
    }                                                                        \
  } while (0)

int get_num_batches(int N, int batch_size)
{
  return (N + batch_size - 1) / batch_size;
}

int get_batch_size(int N, int batch_size, int batch)
{
  int num_batches = get_num_batches(N, batch_size);
  return (batch == num_batches - 1) ? N - batch_size * batch : batch_size;
}

int get_mini_batch_size(int batch_size, int num_procs, int rank)
{
  int mini_batch_size = batch_size / num_procs;
  return rank < batch_size % num_procs ? mini_batch_size + 1 : mini_batch_size;
}

nn_real norms(NeuralNetwork &nn)
{
  nn_real norm_sum = 0;

  for (int i = 0; i < nn.num_layers; ++i)
  {
    norm_sum += arma::accu(arma::square(nn.W[i]));
  }

  return norm_sum;
}

/* CPU implementation.
 * Follow this code to build your GPU code.
 */

// Sigmoid activation
void sigmoid(const arma::Mat<nn_real> &mat, arma::Mat<nn_real> &mat2)
{
  mat2.set_size(mat.n_rows, mat.n_cols);
  ASSERT_MAT_SAME_SIZE(mat, mat2);
  mat2 = 1 / (1 + arma::exp(-mat));
}

// Softmax activation
void softmax(const arma::Mat<nn_real> &mat, arma::Mat<nn_real> &mat2)
{
  mat2.set_size(mat.n_rows, mat.n_cols);
  arma::Mat<nn_real> exp_mat = arma::exp(mat);
  arma::Mat<nn_real> sum_exp_mat = arma::sum(exp_mat, 0);
  mat2 = exp_mat / repmat(sum_exp_mat, mat.n_rows, 1);
}

// feedforward pass
void feedforward(NeuralNetwork &nn, const arma::Mat<nn_real> &X,
                 struct cache &cache)
{
  cache.z.resize(2);
  cache.a.resize(2);

  assert(X.n_rows == nn.W[0].n_cols);
  cache.X = X;
  int N = X.n_cols;

  arma::Mat<nn_real> z1 = nn.W[0] * X + arma::repmat(nn.b[0], 1, N);
  cache.z[0] = z1;

  arma::Mat<nn_real> a1;
  sigmoid(z1, a1);
  cache.a[0] = a1;

  assert(a1.n_rows == nn.W[1].n_cols);
  arma::Mat<nn_real> z2 = nn.W[1] * a1 + arma::repmat(nn.b[1], 1, N);
  cache.z[1] = z2;

  arma::Mat<nn_real> a2;
  softmax(z2, a2);
  cache.a[1] = cache.yc = a2;
}

/*
 * Computes the gradients of the cost w.r.t each param.
 * MUST be called after feedforward since it uses the bpcache.
 * @params y : C x N one-hot column vectors
 * @params bpcache : Output of feedforward.
 * @params bpgrads: Returns the gradients for each param
 */
void backprop(NeuralNetwork &nn, const arma::Mat<nn_real> &y, nn_real reg,
              const struct cache &bpcache, struct grads &bpgrads)
{
  bpgrads.dW.resize(2);
  bpgrads.db.resize(2);
  int N = y.n_cols;

  // std::cout << "backprop " << bpcache.yc << "\n";
  arma::Mat<nn_real> diff = (1.0 / N) * (bpcache.yc - y);
  bpgrads.dW[1] = diff * bpcache.a[0].t() + reg * nn.W[1];
  bpgrads.db[1] = arma::sum(diff, 1);
  arma::Mat<nn_real> da1 = nn.W[1].t() * diff;

  arma::Mat<nn_real> dz1 = da1 % bpcache.a[0] % (1 - bpcache.a[0]);

  bpgrads.dW[0] = dz1 * bpcache.X.t() + reg * nn.W[0];
  bpgrads.db[0] = arma::sum(dz1, 1);
}

/*
 * Computes the Cross-Entropy loss function for the neural network.
 */
nn_real loss(NeuralNetwork &nn, const arma::Mat<nn_real> &yc,
             const arma::Mat<nn_real> &y, nn_real reg)
{
  int N = yc.n_cols;
  nn_real ce_sum = -arma::accu(arma::log(yc.elem(arma::find(y == 1))));

  nn_real data_loss = ce_sum / N;
  nn_real reg_loss = 0.5 * reg * norms(nn);
  nn_real loss = data_loss + reg_loss;
  // std::cout << "Loss: " << loss << "\n";
  return loss;
}

/*
 * Returns a vector of labels for each row vector in the input
 */
void predict(NeuralNetwork &nn, const arma::Mat<nn_real> &X,
             arma::Row<nn_real> &label)
{
  struct cache fcache;
  feedforward(nn, X, fcache);
  label.set_size(X.n_cols);

  for (int i = 0; i < X.n_cols; ++i)
  {
    arma::uword row;
    fcache.yc.col(i).max(row);
    label(i) = row;
  }
}

/*
 * Train the neural network &nn
 */
void train(NeuralNetwork &nn, const arma::Mat<nn_real> &X,
           const arma::Mat<nn_real> &y, nn_real learning_rate, nn_real reg,
           const int epochs, const int batch_size, bool grad_check,
           int print_every, int debug)
{
  int N = X.n_cols;
  int iter = 0;
  int print_flag = 0;

  assert(X.n_cols == y.n_cols);

  int num_batches = get_num_batches(N, batch_size);

  for (int epoch = 0; epoch < epochs; ++epoch)
  {
    int batch_start = 0;
    for (int batch = 0; batch < num_batches; ++batch)
    {
      int last_col = batch_start + get_batch_size(N, batch_size, batch);
      assert(last_col <= X.n_cols);
      assert(last_col <= y.n_cols);
      assert(last_col > batch_start);
      if (batch == num_batches - 1)
      {
        assert(last_col == X.n_cols);
      }
      arma::Mat<nn_real> X_batch = X.cols(batch_start, last_col - 1);
      arma::Mat<nn_real> y_batch = y.cols(batch_start, last_col - 1);

      struct cache bpcache;
      feedforward(nn, X_batch, bpcache);   

      struct grads bpgrads;
      backprop(nn, y_batch, reg, bpcache, bpgrads);
      
      if (print_every > 0 && iter % print_every == 0)
      {
        if (grad_check)
        {
          struct grads numgrads;
          numgrad(nn, X_batch, y_batch, reg, numgrads);
          assert(gradcheck(numgrads, bpgrads));
        }

        std::cout << "Loss at iteration " << iter << " of epoch " << epoch
                  << "/" << epochs << " = "
                  << loss(nn, bpcache.yc, y_batch, reg) << "\n";
      }

      // Gradient descent step
      for (int i = 0; i < nn.W.size(); ++i)
      {
        nn.W[i] -= learning_rate * bpgrads.dW[i];
      }

      for (int i = 0; i < nn.b.size(); ++i)
      {
        nn.b[i] -= learning_rate * bpgrads.db[i];
      }

      /* Debug routine runs only when debug flag is set. If print_every is zero,
         it saves for the first batch of each epoch to avoid saving too many
         large files. Note that for the first time, you have to run debug and
         serial modes together. This will run the following function and write
         out files to the "cpu_save_dir" folder.
         In the later runs (with same parameters), you can use just the debug
         flag to output diff b/w CPU and GPU without running the CPU version
         version. */
      if (print_every <= 0)
      {
        print_flag = batch == 0;
      }
      else
      {
        print_flag = iter % print_every == 0;
      }

      if (debug && print_flag)
      {
        save_cpu_data(nn, iter);
      }

      batch_start = last_col;
      iter++;
    }
  }
}

// for debugging ----
int compareMatrix(nn_real *myC, nn_real *refC, int NI, int NJ)
{
  int fail = 0;

  arma::Mat<nn_real> mysol = arma::Mat<nn_real>(myC, NI, NJ, false);
  arma::Mat<nn_real> refsol = arma::Mat<nn_real>(refC, NI, NJ, false);

  nn_real reldiff = arma::norm(mysol - refsol, "inf") / arma::norm(refsol, "inf");
  nn_real mysol_norm = arma::norm(mysol, "inf");

  if (isnan(mysol_norm) || isinf(mysol_norm))
    std::cout << "Inf or Nan norm of my solution" << std::endl;
  assert(!isnan(mysol_norm));
  assert(!isinf(mysol_norm));

  
  nn_real *ptr1 = mysol.memptr();
  nn_real *ptr2 = refsol.memptr();
  for (int i = 0; i < mysol.n_rows; i+=5)
    for (int j = 0; j < mysol.n_cols; j++)
    {
      //std::cout << "my sol " << ptr1[i*(mysol.n_cols) + j] << "  should be " << ptr2[i*(mysol.n_cols) + j] << std::endl;
    }


  if (reldiff > 1e-10)
  {
    fail = 1;
  }

  // Print results
  if (fail)
  {
    std::cout << "Matricies are not within the tolerance. Rel diff = "
              << reldiff << std::endl;
  }
  else
  {
    std::cout << "Matrix matched with reference successfully! Rel diff = "
              << reldiff << std::endl;
  }

  return fail;
}


/*
 * Allocate memory for weights W and biases b on the GPU
 * and copy them from CPU to GPU
 */
void allocateAndCopyToGPU(gpu_nn &device_nn, gpu_cache &device_cache, gpu_grads &device_grads,
             const arma::Mat<nn_real> &X, const arma::Mat<nn_real> &y, NeuralNetwork &nn, const int batch_size)
{
  const int alloc_N = 21;
  cudaError_t error[alloc_N];
  for (int p = 0; p < alloc_N; p++)
      error[p] = cudaSuccess;

  // W, b, all data (X, y)
  error[0] = cudaMalloc((void **)&device_nn.W1_d, sizeof(nn_real) * nn.H[1] * nn.H[0]);
  error[1] = cudaMalloc((void **)&device_nn.W2_d, sizeof(nn_real) * nn.H[2] * nn.H[1]);
  error[2] = cudaMalloc((void **)&device_nn.b1_d, sizeof(nn_real) * nn.H[1]);
  error[3] = cudaMalloc((void **)&device_nn.b2_d, sizeof(nn_real) * nn.H[2]);
  error[4] = cudaMalloc((void **)&device_nn.all_X_d, sizeof(nn_real) * X.n_rows * X.n_cols);
  error[5] = cudaMalloc((void **)&device_nn.all_y_d, sizeof(nn_real) * y.n_rows * y.n_cols);

  // X_batch, y_batch, a, z, yc
  error[6] = cudaMalloc((void **)&device_cache.X_d, sizeof(nn_real) * nn.H[0] * batch_size);
  error[7] = cudaMalloc((void **)&device_cache.y_d, sizeof(nn_real) * nn.H[2] * batch_size);
  
  error[10] = cudaMalloc((void **)&device_cache.z1_d, sizeof(nn_real) * nn.H[1] * batch_size);
  error[11] = cudaMalloc((void **)&device_cache.z2_d, sizeof(nn_real) * nn.H[2] * batch_size);
  error[12] = cudaMalloc((void **)&device_cache.a1_d, sizeof(nn_real) * nn.H[1] * batch_size);
  error[13] = cudaMalloc((void **)&device_cache.yc_d, sizeof(nn_real) * nn.H[2] * batch_size);

  error[14] = cudaMalloc((void **)&device_cache.diff_y_d, sizeof(nn_real) * nn.H[2] * batch_size);

  // grads
  error[15] = cudaMalloc((void **)&device_grads.dW1_d, sizeof(nn_real) * nn.H[1] * nn.H[0]);
  error[16] = cudaMalloc((void **)&device_grads.dW2_d, sizeof(nn_real) * nn.H[2] * nn.H[1]);
  error[17] = cudaMalloc((void **)&device_grads.db1_d, sizeof(nn_real) * nn.H[1]);
  error[18] = cudaMalloc((void **)&device_grads.db2_d, sizeof(nn_real) * nn.H[2]);
  error[19] = cudaMalloc((void **)&device_grads.da1_d, sizeof(nn_real) * nn.H[1] * batch_size);
  error[20] = cudaMalloc((void **)&device_grads.dz1_d, sizeof(nn_real) * nn.H[1] * batch_size);

  // Check for allocation failure
  for (int p = 0; p < alloc_N; p++)
      if (error[p] != cudaSuccess) std::cout << "Failed to allocate CUDA memory for W and b" << std::endl;

  // Copy the data back from CPU to GPU
  cudaMemcpy(device_nn.W1_d, nn.W[0].memptr(), sizeof(nn_real) * nn.H[1] * nn.H[0], cudaMemcpyHostToDevice);
  cudaMemcpy(device_nn.W2_d, nn.W[1].memptr(), sizeof(nn_real) * nn.H[2] * nn.H[1], cudaMemcpyHostToDevice);
  cudaMemcpy(device_nn.b1_d, nn.b[0].memptr(), sizeof(nn_real) * nn.H[1], cudaMemcpyHostToDevice);
  cudaMemcpy(device_nn.b2_d, nn.b[1].memptr(), sizeof(nn_real) * nn.H[2], cudaMemcpyHostToDevice);
  cudaMemcpy(device_nn.all_X_d, X.memptr(), sizeof(nn_real) * X.n_rows * X.n_cols, cudaMemcpyHostToDevice);
  cudaMemcpy(device_nn.all_y_d, y.memptr(), sizeof(nn_real) * y.n_rows * y.n_cols, cudaMemcpyHostToDevice);
}

void allocateAndCopyToGPU_MPI(gpu_nn &device_nn, gpu_cache &device_cache, gpu_grads &device_grads,
             const arma::Mat<nn_real> &X, const arma::Mat<nn_real> &y, NeuralNetwork &nn, const int mini_batch_size)
{
  const int alloc_N = 21;
  cudaError_t error[alloc_N];
  for (int p = 0; p < alloc_N; p++)
      error[p] = cudaSuccess;

  // W, b, all data (X, y)
  error[0] = cudaMalloc((void **)&device_nn.W1_d, sizeof(nn_real) * nn.H[1] * nn.H[0]);
  error[1] = cudaMalloc((void **)&device_nn.W2_d, sizeof(nn_real) * nn.H[2] * nn.H[1]);
  error[2] = cudaMalloc((void **)&device_nn.b1_d, sizeof(nn_real) * nn.H[1]);
  error[3] = cudaMalloc((void **)&device_nn.b2_d, sizeof(nn_real) * nn.H[2]);

  // X_batch, y_batch, a, z, yc
  error[6] = cudaMalloc((void **)&device_cache.X_d, sizeof(nn_real) * nn.H[0] * mini_batch_size);
  error[7] = cudaMalloc((void **)&device_cache.y_d, sizeof(nn_real) * nn.H[2] * mini_batch_size);
  
  error[10] = cudaMalloc((void **)&device_cache.z1_d, sizeof(nn_real) * nn.H[1] * mini_batch_size);
  error[11] = cudaMalloc((void **)&device_cache.z2_d, sizeof(nn_real) * nn.H[2] * mini_batch_size);
  error[12] = cudaMalloc((void **)&device_cache.a1_d, sizeof(nn_real) * nn.H[1] * mini_batch_size);
  error[13] = cudaMalloc((void **)&device_cache.yc_d, sizeof(nn_real) * nn.H[2] * mini_batch_size);

  error[14] = cudaMalloc((void **)&device_cache.diff_y_d, sizeof(nn_real) * nn.H[2] * mini_batch_size);

  // grads
  error[15] = cudaMalloc((void **)&device_grads.dW1_d, sizeof(nn_real) * nn.H[1] * nn.H[0]);
  error[16] = cudaMalloc((void **)&device_grads.dW2_d, sizeof(nn_real) * nn.H[2] * nn.H[1]);
  error[17] = cudaMalloc((void **)&device_grads.db1_d, sizeof(nn_real) * nn.H[1]);
  error[18] = cudaMalloc((void **)&device_grads.db2_d, sizeof(nn_real) * nn.H[2]);
  error[19] = cudaMalloc((void **)&device_grads.da1_d, sizeof(nn_real) * nn.H[1] * mini_batch_size);
  error[20] = cudaMalloc((void **)&device_grads.dz1_d, sizeof(nn_real) * nn.H[1] * mini_batch_size);

  // Check for allocation failure
  for (int p = 0; p < alloc_N; p++)
      if (error[p] != cudaSuccess) std::cout << "Failed to allocate CUDA memory for W and b" << std::endl;

  // Copy the data back from CPU to GPU
  cudaMemcpy(device_nn.W1_d, nn.W[0].memptr(), sizeof(nn_real) * nn.H[1] * nn.H[0], cudaMemcpyHostToDevice);
  cudaMemcpy(device_nn.W2_d, nn.W[1].memptr(), sizeof(nn_real) * nn.H[2] * nn.H[1], cudaMemcpyHostToDevice);
  cudaMemcpy(device_nn.b1_d, nn.b[0].memptr(), sizeof(nn_real) * nn.H[1], cudaMemcpyHostToDevice);
  cudaMemcpy(device_nn.b2_d, nn.b[1].memptr(), sizeof(nn_real) * nn.H[2], cudaMemcpyHostToDevice);
}

void allocateAndCopyToCPU_MPI(cpu_cache &host_cache, NeuralNetwork &nn, const arma::Mat<nn_real> &X, 
                            const arma::Mat<nn_real> &y, const int mini_batch_size, int rank)
{
    if (rank == 0)
    {
      host_cache.all_X_h = (nn_real *)malloc(sizeof(nn_real) * X.n_rows * X.n_cols);
      host_cache.all_y_h = (nn_real *)malloc(sizeof(nn_real) * y.n_rows * y.n_cols);

      memcpy(host_cache.all_X_h, X.memptr(), sizeof(nn_real) * X.n_rows * X.n_cols);
      memcpy(host_cache.all_y_h, y.memptr(), sizeof(nn_real) * y.n_rows * y.n_cols);
    }
    
    host_cache.X_h = (nn_real *)malloc(sizeof(nn_real) * nn.H[0] * mini_batch_size);
    host_cache.y_h = (nn_real *)malloc(sizeof(nn_real) * nn.H[2] * mini_batch_size);

    host_cache.dW1_partial_h = (nn_real *)malloc(sizeof(nn_real) * nn.H[1] * nn.H[0]);
    host_cache.dW2_partial_h = (nn_real *)malloc(sizeof(nn_real) * nn.H[2] * nn.H[1]);
    host_cache.db1_partial_h = (nn_real *)malloc(sizeof(nn_real) * nn.H[1]);
    host_cache.db2_partial_h = (nn_real *)malloc(sizeof(nn_real) * nn.H[2]);

    host_cache.dW1_total_h = (nn_real *)malloc(sizeof(nn_real) * nn.H[1] * nn.H[0]);
    host_cache.dW2_total_h = (nn_real *)malloc(sizeof(nn_real) * nn.H[2] * nn.H[1]);
    host_cache.db1_total_h = (nn_real *)malloc(sizeof(nn_real) * nn.H[1]);
    host_cache.db2_total_h = (nn_real *)malloc(sizeof(nn_real) * nn.H[2]);
}

void freeMemoryOnGPU(gpu_nn &device_nn, gpu_cache &device_cache, gpu_grads &device_grads)
{
  // W and b
  cudaFree(device_nn.W1_d);
  cudaFree(device_nn.W2_d);
  cudaFree(device_nn.b1_d);
  cudaFree(device_nn.b2_d);

  // X_batch, yc_batch, a, z
  cudaFree(device_cache.X_d);
  cudaFree(device_cache.y_d);  
  cudaFree(device_cache.z1_d);
  cudaFree(device_cache.z2_d);
  cudaFree(device_cache.a1_d);
  cudaFree(device_cache.yc_d);
  cudaFree(device_cache.diff_y_d);

  // grads
  cudaFree(device_grads.dW1_d);
  cudaFree(device_grads.dW2_d);
  cudaFree(device_grads.db1_d);
  cudaFree(device_grads.db2_d);
  cudaFree(device_grads.da1_d);
  cudaFree(device_grads.dz1_d);
}

void freeMemoryOnCPU(cpu_cache &host_cache, int rank)
{
  if (rank == 0)
  {
    free(host_cache.all_X_h);
    free(host_cache.all_y_h);
  }
  free(host_cache.X_h);
  free(host_cache.y_h);

  free(host_cache.dW1_partial_h);
  free(host_cache.dW2_partial_h);
  free(host_cache.db1_partial_h);
  free(host_cache.db2_partial_h);

  free(host_cache.dW1_total_h);
  free(host_cache.dW2_total_h);
  free(host_cache.db1_total_h);
  free(host_cache.db2_total_h);
}


// feedforward pass
void feedforward_device(gpu_nn &device_nn, gpu_cache &device_cache, NeuralNetwork &nn, int mini_batch_size)
{ 
  // z(1) = W(1) * Xbatch + b(1)
  myABCD_GEMM(device_nn.W1_d, device_cache.X_d, device_nn.b1_d, device_cache.z1_d, 1, 1, nn.H[1], mini_batch_size, nn.H[0]);

  // a(1) = sigmoid(z(1))
  my_sigmoid(device_cache.z1_d, device_cache.a1_d, nn.H[1] * mini_batch_size);

  // z(2) = W(2) * a(1) + b(2)
  myABCD_GEMM(device_nn.W2_d, device_cache.a1_d, device_nn.b2_d, device_cache.z2_d, 1, 1, nn.H[2], mini_batch_size, nn.H[1]);

  // yc = a(2) = sigmoid(z(2))
  my_softmax(device_cache.z2_d, device_cache.yc_d, mini_batch_size, nn.H[2]);
}

void backprop_device(NeuralNetwork &nn, nn_real reg, gpu_nn &device_nn, gpu_cache &device_cache, 
                        gpu_grads &device_grads, int mini_batch_size, int N)
{
  // diff = 1 / N * (yc - y)
  matrixAdd(device_cache.yc_d, device_cache.y_d, device_cache.diff_y_d, 1.0 / N, - 1.0 / N, nn.H[2], mini_batch_size);

  // dW2 = (yc - y) * a(1)^T + reg * W(2)
  myABtCD_GEMM(device_cache.diff_y_d, device_cache.a1_d, device_nn.W2_d, device_grads.dW2_d, 1, reg, nn.H[2], nn.H[1], mini_batch_size);

  // db2 = sum(yc - y) over the batch_size
  my_reduce_A(device_cache.diff_y_d, device_grads.db2_d, nn.H[2], mini_batch_size);

  // da1 = W(2)^T * (yc - y)
  myAtBCD_GEMM(device_nn.W2_d, device_cache.diff_y_d, device_grads.da1_d, device_grads.da1_d, 1, 0, nn.H[1], mini_batch_size, nn.H[2]);

  // dz1 = da1 % a1 % (1 - a1)
  myABCD_elementwise(device_grads.da1_d, device_cache.a1_d, device_cache.a1_d, device_grads.dz1_d, 1, nn.H[1], mini_batch_size);

  // dW1 = dz1 * X_batch^T + reg * W(1)
  myABtCD_GEMM(device_grads.dz1_d, device_cache.X_d, device_nn.W1_d, device_grads.dW1_d, 1, reg, nn.H[1], nn.H[0], mini_batch_size);

  // db1 = sum(dz1) over the batch_size
  my_reduce_A(device_grads.dz1_d, device_grads.db1_d, nn.H[1], mini_batch_size);
}

void gradient_descent_device(gpu_nn &device_nn, gpu_grads &device_grads, nn_real learning_rate, 
                        NeuralNetwork &nn)
{

  // update W1 = W1 - learning_rate * dW1
  matrixAdd(device_nn.W1_d, device_grads.dW1_d, device_nn.W1_d, 1, -learning_rate, nn.H[1], nn.H[0]);

  // update W2 = W2 - learning_rate * dW2
  matrixAdd(device_nn.W2_d, device_grads.dW2_d, device_nn.W2_d, 1, -learning_rate, nn.H[2], nn.H[1]);

  // update b1 = b1 - learning_rate * db1
  matrixAdd(device_nn.b1_d, device_grads.db1_d, device_nn.b1_d, 1, -learning_rate, nn.H[1], 1);

  // update b2 = b2 - learning_rate * db2
  matrixAdd(device_nn.b2_d, device_grads.db2_d, device_nn.b2_d, 1, -learning_rate, nn.H[2], 1);
}

void updateNeuralNetworkOnCPU(gpu_nn &device_nn, NeuralNetwork &nn)
{
      cudaMemcpy(nn.W[1].memptr(), device_nn.W2_d, sizeof(nn_real) * nn.H[2] * nn.H[1], cudaMemcpyDeviceToHost);
      cudaMemcpy(nn.b[1].memptr(), device_nn.b2_d, sizeof(nn_real) * nn.H[2], cudaMemcpyDeviceToHost);
      cudaMemcpy(nn.W[0].memptr(), device_nn.W1_d, sizeof(nn_real) * nn.H[1] * nn.H[0], cudaMemcpyDeviceToHost);
      cudaMemcpy(nn.b[0].memptr(), device_nn.b1_d, sizeof(nn_real) * nn.H[1], cudaMemcpyDeviceToHost);
}

/*
 * Train the neural network &nn of rank 0 in parallel. Your MPI implementation
 * should mainly be in this function.
 */
void parallel_train(NeuralNetwork &nn, const arma::Mat<nn_real> &X,
                    const arma::Mat<nn_real> &y, nn_real learning_rate,
                    std::ofstream &error_file, nn_real reg, const int epochs,
                    const int batch_size, int print_every, int debug)
{
  assert(learning_rate > 0);
  assert(reg >= 0);
  assert(epochs >= 0);
  assert(batch_size > 0);

  int rank, num_procs;
  MPI_SAFE_CALL(MPI_Comm_size(MPI_COMM_WORLD, &num_procs));
  MPI_SAFE_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

  if (rank == 0)
  {
    assert(X.n_cols > 0);
    assert(X.n_rows == IMAGE_SIZE);
    assert(y.n_cols == X.n_cols);
    assert(y.n_rows == NUM_CLASSES);
    assert(nn.H[0] == IMAGE_SIZE);
    assert(nn.H[2] == NUM_CLASSES);
  }

  int N = (rank == 0) ? X.n_cols : 0;

  MPI_SAFE_CALL(MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD));

  assert(N > 0);

  int print_flag = 0;

  // Data sets
  const int num_batches = get_num_batches(N, batch_size);
  int mini_batch_size_alloc;
  {
    const int max_batch_size = batch_size;
    mini_batch_size_alloc = max_batch_size / num_procs + 1;
  }

  struct cpu_cache host_cache;
  allocateAndCopyToCPU_MPI(host_cache, nn, X, y, mini_batch_size_alloc, rank);

  struct gpu_nn device_nn;
  struct gpu_cache device_cache;
  struct gpu_grads device_grads;
  allocateAndCopyToGPU_MPI(device_nn, device_cache, device_grads, X, y, nn, mini_batch_size_alloc);


  /* iter is a variable used to manage debugging. It increments in the inner
     loop and therefore goes from 0 to epochs*num_batches */
  int iter = 0;

  for (int epoch = 0; epoch < epochs; ++epoch)
  {
    int batch_start = 0;

    for (int batch = 0; batch < num_batches; ++batch)
    {   
      // get current batch_size
      int cur_batch_size = get_batch_size(N,  batch_size, batch);
      int cur_mini_batch_size = get_mini_batch_size(cur_batch_size, num_procs, rank);

      // pointer to current batch
      nn_real *X_batch;
      nn_real *y_batch;
      if (rank == 0)
      {
          X_batch = host_cache.all_X_h + (batch_start)*nn.H[0];
          y_batch = host_cache.all_y_h + (batch_start)*nn.H[2];
      }
      
      // Scatter the data if there are more than one process
      if (num_procs > 1)
      {
        if (batch_size % num_procs == 0) // Case of 2, 4 GPUs
        {
            MPI_SAFE_CALL(MPI_Scatter(X_batch, cur_mini_batch_size * nn.H[0], MPI_FP,
                host_cache.X_h, cur_mini_batch_size * nn.H[0], MPI_FP, 0, MPI_COMM_WORLD));
            MPI_SAFE_CALL(MPI_Scatter(y_batch, cur_mini_batch_size * nn.H[2], MPI_FP,
                host_cache.y_h, cur_mini_batch_size * nn.H[2], MPI_FP, 0, MPI_COMM_WORLD));
        }
        else // Case of 3 GPUs
        {
            int *imagecnt, *displsX, *displsy,  *scountsX, *scountsy;
            displsX = (int *)malloc(num_procs*sizeof(int));
            displsy = (int *)malloc(num_procs*sizeof(int));
            scountsX = (int *)malloc(num_procs*sizeof(int));
            scountsy = (int *)malloc(num_procs*sizeof(int));
            imagecnt = (int *)malloc(num_procs*sizeof(int));

            int uneven_imag = cur_batch_size % num_procs;
            for(uint k = 0; k < num_procs; k++){
              imagecnt[k] = cur_batch_size / num_procs;
              if (k < uneven_imag) 
                imagecnt[k]++;
            }

            for(uint k = 0; k < num_procs; k++)
            {
              scountsX[k] = imagecnt[k] * nn.H[0];
              scountsy[k] = imagecnt[k] * nn.H[2];
            }

            displsX[0] = 0; displsy[0] = 0;            
            for (uint i = 1; i < num_procs; i++)
            {
              displsX[i] = displsX[i-1] + scountsX[i-1];  
              displsy[i] = displsy[i-1] + scountsy[i-1];  
            }   
            
            MPI_Scatterv(X_batch, scountsX, displsX, MPI_FP, host_cache.X_h, mini_batch_size_alloc * nn.H[0], MPI_FP, 0, MPI_COMM_WORLD);
            MPI_Scatterv(y_batch, scountsy, displsy, MPI_FP, host_cache.y_h, mini_batch_size_alloc * nn.H[2], MPI_FP, 0, MPI_COMM_WORLD);
        }
      }
      else // Case of 1 GPU
      {
         host_cache.X_h = X_batch;
         host_cache.y_h = y_batch;
      }

      //std::cout << "passed MPI_scatter" << std::endl;

      // copy the mini batch from CPU to GPU
      cudaMemcpy(device_cache.X_d, host_cache.X_h, sizeof(nn_real) * cur_mini_batch_size * nn.H[0], cudaMemcpyHostToDevice);
      cudaMemcpy(device_cache.y_d, host_cache.y_h, sizeof(nn_real) * cur_mini_batch_size * nn.H[2], cudaMemcpyHostToDevice);

      // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
      //                          FEED FORWARD                            //
      // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
      feedforward_device(device_nn, device_cache, nn, cur_mini_batch_size);

      // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
      //                         BACK PROPAGATE                           //
      // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
      backprop_device(nn, reg / num_procs, device_nn, device_cache, device_grads, cur_mini_batch_size, cur_batch_size);

      if (num_procs > 1)
      {
        // copy back the gradients from GPU to CPU
        cudaMemcpy(host_cache.dW2_partial_h, device_grads.dW2_d, sizeof(nn_real) * nn.H[2] * nn.H[1], cudaMemcpyDeviceToHost);
        cudaMemcpy(host_cache.db2_partial_h, device_grads.db2_d, sizeof(nn_real) * nn.H[2], cudaMemcpyDeviceToHost);
        cudaMemcpy(host_cache.dW1_partial_h, device_grads.dW1_d, sizeof(nn_real) * nn.H[1] * nn.H[0], cudaMemcpyDeviceToHost);
        cudaMemcpy(host_cache.db1_partial_h, device_grads.db1_d, sizeof(nn_real) * nn.H[1], cudaMemcpyDeviceToHost);

        // MPI Allreduce
        MPI_SAFE_CALL(MPI_Allreduce(host_cache.dW2_partial_h, host_cache.dW2_total_h, nn.H[2] * nn.H[1], MPI_FP, MPI_SUM, MPI_COMM_WORLD));
        MPI_SAFE_CALL(MPI_Allreduce(host_cache.db2_partial_h, host_cache.db2_total_h, nn.H[2], MPI_FP, MPI_SUM, MPI_COMM_WORLD));
        MPI_SAFE_CALL(MPI_Allreduce(host_cache.dW1_partial_h, host_cache.dW1_total_h, nn.H[1] * nn.H[0], MPI_FP, MPI_SUM, MPI_COMM_WORLD));
        MPI_SAFE_CALL(MPI_Allreduce(host_cache.db1_partial_h, host_cache.db1_total_h, nn.H[1], MPI_FP, MPI_SUM, MPI_COMM_WORLD)); 
      
        // Copy the total gradients from CPU to GPU
        cudaMemcpy(device_grads.dW2_d, host_cache.dW2_total_h, sizeof(nn_real) * nn.H[2] * nn.H[1], cudaMemcpyHostToDevice);
        cudaMemcpy(device_grads.db2_d, host_cache.db2_total_h, sizeof(nn_real) * nn.H[2], cudaMemcpyHostToDevice);
        cudaMemcpy(device_grads.dW1_d, host_cache.dW1_total_h, sizeof(nn_real) * nn.H[1] * nn.H[0], cudaMemcpyHostToDevice);
        cudaMemcpy(device_grads.db1_d, host_cache.db1_total_h, sizeof(nn_real) * nn.H[1], cudaMemcpyHostToDevice);
      }
      
      // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
      //                    GRADIENT DESCENT STEP                         //
      // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
      gradient_descent_device(device_nn, device_grads, learning_rate, nn);

      // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
      //                    POST-PROCESS OPTIONS                          //
      // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //


      if (print_every <= 0)
      {
        print_flag = batch == 0;
      }
      else
      {
        print_flag = iter % print_every == 0;
      }

      /* Following debug routine assumes that you have already updated the arma
         matrices in the NeuralNetwork nn.  */
      if (debug && rank == 0 && print_flag)
      {
        // Copy data back to the CPU
        updateNeuralNetworkOnCPU(device_nn, nn);
        save_gpu_error(nn, iter, error_file);
      }
      batch_start += cur_batch_size;
      iter++;
    }
  }

  // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
  //                  Update Neural Network on CPU                    //
  // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
  if (rank == 0) updateNeuralNetworkOnCPU(device_nn, nn);


  // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
  //                    Free memory allocations                       //
  // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
  //freeMemoryOnCPU(host_cache, rank);
  freeMemoryOnGPU(device_nn, device_cache, device_grads);
}
