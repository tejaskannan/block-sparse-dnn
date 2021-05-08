#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "bsmm.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "cuda_runtime.h"
#include "cuda.h"


using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

template<class T>
struct shared_memory
{
  __device__ inline operator T *()
  {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }

  __device__ inline operator const T *() const
  {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }
};

template <typename index_type>
__device__ index_type round_up_to_power_of_two (index_type v)
{
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;

  return v;
}

// CUDA kernel based on https://medium.com/gpgpu/block-sparse-matrix-vector-multiplication-with-cuda-4e616b30267
template <typename data_type, typename index_type>
__global__ void bcsr_spmv_kernel_column_by_column(
  const index_type bs,
  const index_type * __restrict__ col_ids,
  const index_type * __restrict__ row_ptr,
  const data_type * __restrict__ data,
  const data_type * __restrict__ x,
  data_type * __restrict__ y)
{
  const index_type idx = blockIdx.x * blockDim.x + threadIdx.x;
  const index_type lane = idx % 32;
  const index_type block_row = idx / 32; ///< Warp per block row
  const index_type first_block = row_ptr[block_row];
  const index_type last_block = row_ptr[block_row + 1];

  index_type col = first_block * bs + lane / bs;
  index_type r = lane % bs;

  data_type *partial_sums = shared_memory<data_type> (); ///< Size is equal to blockDim.x * sizeof(data_type)

  data_type local_out = 0.0;

  for (; col < last_block * bs; col += 32 / bs)
    {
      const index_type block = col / bs;
      const index_type c = col % bs;

      const data_type value = data[block * bs * bs + c * bs + r];
      const data_type x_value = x[col_ids[block] * bs + c];
      local_out += x_value * value;
    }

  partial_sums[threadIdx.x] = local_out;

  for (index_type stride = round_up_to_power_of_two((32 / bs) / 2); stride > 0; stride /= 2)
    {
      __syncthreads ();
      if ((lane < stride * bs) && ((threadIdx.x + stride * bs) < 32))
        partial_sums[threadIdx.x] += partial_sums[threadIdx.x + stride * bs];
    }

  if (lane < bs)
    y[block_row * bs + lane] = partial_sums[threadIdx.x];
}

template<typename DataType, typename IndexType>
void BCSRMatMulFunctor<GPUDevice, DataType, IndexType>::operator()(const GPUDevice& d, int block_size, IndexType* col_ids, IndexType* row_ptr, DataType* blocks, DataType* dense, DataType* out) {
        int block_count = 1024;
        int thread_per_block = 20;

        bcsr_spmv_kernel_column_by_column<DataType, IndexType><<<block_count, thread_per_block, 0, d.stream()>>>(block_size, col_ids, row_ptr, blocks, dense, out);
    }


// Explicitly instantiate functors for the types of OpKernels registered.
template struct BCSRMatMulFunctor<GPUDevice, float, uint64>;

#endif  // GOOGLE_CUDA
