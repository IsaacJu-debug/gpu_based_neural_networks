#ifndef GPU_UTIL_H_
#define GPU_UTIL_H_

struct event_pair
{
  cudaEvent_t start;
  cudaEvent_t end;
};

inline void check_launch(const char *kernel_name)
{
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();

  if (err != cudaSuccess)
  {
    std::cerr << "error in " << kernel_name << " kernel" << std::endl;
    std::cerr << "error was: " << cudaGetErrorString(err) << std::endl;
    exit(1);
  }
}

inline void start_timer(event_pair *p)
{
  cudaEventCreate(&p->start);
  cudaEventCreate(&p->end);
  cudaEventRecord(p->start, 0);
}

inline double stop_timer(event_pair *p)
{
  cudaEventRecord(p->end, 0);
  cudaEventSynchronize(p->end);

  float elapsed_time;
  cudaEventElapsedTime(&elapsed_time, p->start, p->end);
  cudaEventDestroy(p->start);
  cudaEventDestroy(p->end);
  return elapsed_time;
}

#endif