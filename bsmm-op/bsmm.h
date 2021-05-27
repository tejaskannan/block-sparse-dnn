#ifndef BSMM_H
#define BSMM_H
#include <unsupported/Eigen/CXX11/Tensor>

template <typename Device, typename DataType, typename IndexType>
struct BCSRMatMulFunctor
{
    void operator()(const Device& d, int block_size, IndexType* col_ids, IndexType* row_ptr, DataType* blocks, DataType* dense, DataType* out);
};

#if GOOGLE_CUDA
template<typename DataType, typename IndexType>
struct BCSRMatMulFunctor<Eigen::GpuDevice, DataType, IndexType>
{
    void operator()(const Eigen::GpuDevice& d, int block_size, IndexType* col_ids, IndexType* row_ptr, DataType* blocks, DataType* dense, DataType* out);
};
#endif // GOOGLE_CUDA

template <typename Device, typename DataType, typename IndexType>
struct BRCMatMulFunctor
{
    void operator()(const Device& d, int block_size, IndexType units, IndexType* pattern, DataType* blocks, DataType* dense, DataType* out);
};

#if GOOGLE_CUDA
template<typename DataType, typename IndexType>
struct BRCMatMulFunctor<Eigen::GpuDevice, DataType, IndexType>
{
    void operator()(const Eigen::GpuDevice& d, int block_size, IndexType units, IndexType* input_indices, DataType* blocks, DataType* dense, DataType* out);
};
#endif // GOOGLE_CUDA


#endif // BSMM_H