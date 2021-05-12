#include "bsmm.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <stdexcept>

using namespace tensorflow;
using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("BCSRMatMul")
    .Attr("DataType: numbertype")
    .Attr("IndexType: numbertype")
    .Attr("block_size: int")
    .Input("col_ids : IndexType")
    .Input("row_ptr : IndexType")
    .Input("blocks : DataType")
    .Input("dense : DataType")
    .Output("sparse_times_dense : DataType")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(3));
      return Status::OK();
    });

template<typename DataType, typename IndexType>
struct BCSRMatMulFunctor<CPUDevice, DataType, IndexType>
{
    void operator()(const CPUDevice& d, int block_size, const IndexType* col_ids, const IndexType* row_ptr, const DataType* blocks, const DataType* dense, DataType* out)
    {
       //throw std::runtime_error("CPU version not implemented");
    }
};

template <typename Device, typename DataType, typename IndexType>
class BCSRMatMul : public OpKernel {
    public:
    explicit BCSRMatMul(OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("block_size", &block_size_));
    }

    void Compute(OpKernelContext* context) override {
        const Tensor& col_ids = context->input(0);
        const Tensor& row_ptr = context->input(1);
        const Tensor& blocks = context->input(2);
        const Tensor& dense = context->input(3);

        Tensor* output_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, dense.shape(), &output_tensor));

        BCSRMatMulFunctor<Device, DataType, IndexType>()(
            context->eigen_device<Device>(),
            block_size_,
            col_ids.flat<IndexType>().data(),
            row_ptr.flat<IndexType>().data(),
            blocks.flat<DataType>().data(),
            dense.flat<DataType>().data(),
            output_tensor->flat<DataType>().data());
        
    }
    private:
        IndexType block_size_;
};

#define REGISTER_CPU_BCSR(DataType, IndexType)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("BCSRMatMul").Device(DEVICE_CPU).TypeConstraint<DataType>("DataType").TypeConstraint<IndexType>("IndexType"), \
      BCSRMatMul<CPUDevice, DataType, IndexType>);
REGISTER_CPU_BCSR(float, tensorflow::uint64);
REGISTER_CPU_BCSR(double, tensorflow::int64);

#ifdef GOOGLE_CUDA
#define REGISTER_GPU_BCSR(DataType, IndexType)                                          \
  extern template class BCSRMatMulFunctor<GPUDevice, DataType, IndexType>;            \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("BCSRMatMul").Device(DEVICE_GPU).TypeConstraint<DataType>("DataType").TypeConstraint<IndexType>("IndexType"), \
      BCSRMatMul<GPUDevice, DataType, IndexType>);
REGISTER_GPU_BCSR(float, tensorflow::uint64);
REGISTER_GPU_BCSR(double, tensorflow::int64);
#endif  // GOOGLE_CUDA


REGISTER_OP("BCSRMatMulNA")
    .Attr("DataType: numbertype")
    .Attr("IndexType: numbertype")
    .Input("block_size: IndexType")
    .Input("col_ids : IndexType")
    .Input("row_ptr : IndexType")
    .Input("blocks : DataType")
    .Input("dense : DataType")
    .Output("sparse_times_dense : DataType")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(3));
      return Status::OK();
    });

template <typename Device, typename DataType, typename IndexType>
class BCSRMatMulNA : public OpKernel {
    public:
    explicit BCSRMatMulNA(OpKernelConstruction* context) : OpKernel(context) {
    }

    void Compute(OpKernelContext* context) override {
        const Tensor& block_size = context->input(0);
        const Tensor& col_ids = context->input(1);
        const Tensor& row_ptr = context->input(2);
        const Tensor& blocks = context->input(3);
        const Tensor& dense = context->input(4);

        Tensor* output_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, dense.shape(), &output_tensor));

        BCSRMatMulFunctor<Device, DataType, IndexType>()(
            context->eigen_device<Device>(),
            block_size.flat<IndexType>().data()[0],
            col_ids.flat<IndexType>().data(),
            row_ptr.flat<IndexType>().data(),
            blocks.flat<DataType>().data(),
            dense.flat<DataType>().data(),
            output_tensor->flat<DataType>().data());
        
    }
};

#define REGISTER_CPU_BCSRNA(DataType, IndexType)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("BCSRMatMulNA").Device(DEVICE_CPU).TypeConstraint<DataType>("DataType").TypeConstraint<IndexType>("IndexType"), \
      BCSRMatMulNA<CPUDevice, DataType, IndexType>);
REGISTER_CPU_BCSRNA(float, tensorflow::uint64);
REGISTER_CPU_BCSRNA(double, tensorflow::int64);
REGISTER_CPU_BCSRNA(double, tensorflow::int32);

#ifdef GOOGLE_CUDA
#define REGISTER_GPU_BCSRNA(DataType, IndexType)                                          \
  extern template class BCSRMatMulFunctor<GPUDevice, DataType, IndexType>;            \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("BCSRMatMulNA").Device(DEVICE_GPU).TypeConstraint<DataType>("DataType").TypeConstraint<IndexType>("IndexType"), \
      BCSRMatMulNA<GPUDevice, DataType, IndexType>);
REGISTER_GPU_BCSRNA(float, tensorflow::uint64);
REGISTER_GPU_BCSRNA(double, tensorflow::int64);
REGISTER_GPU_BCSRNA(double, tensorflow::int32);
#endif  // GOOGLE_CUDA