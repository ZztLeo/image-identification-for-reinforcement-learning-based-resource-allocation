/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// See docs in ../ops/nn_ops.cc.
#ifdef INTEL_MKL
#define EIGEN_USE_THREADS
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/mkl_pooling_ops_common.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/mkl_util.h"
#include "tensorflow/core/util/padding.h"

#ifndef INTEL_MKL_ML
#include <algorithm>
#include "mkldnn.hpp"
using mkldnn::algorithm;
using mkldnn::engine;
using mkldnn::error;
using mkldnn::memory;
using mkldnn::padding_kind;
using mkldnn::pooling_backward;
using mkldnn::pooling_forward;
using mkldnn::prop_kind;
#endif

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

// MKL-DNN is now default. MKL-ML must be specified explicitly.
#ifdef INTEL_MKL_ML

// An implementation of MaxPooling (forward).
template <typename Device, typename T>
class MklMaxPoolingOp : public OpKernel {
 public:
  explicit MklMaxPoolingOp(OpKernelConstruction* context) : OpKernel(context) {
    string data_format;

    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
    OP_REQUIRES(context, ksize_.size() == 4,
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 4,
                errors::InvalidArgument("Sliding window stride field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    OP_REQUIRES(context, ksize_[0] == 1 && stride_[0] == 1,
                errors::Unimplemented("Pooling is not yet supported on the "
                                      "batch dimension."));

    workspace_enabled_ = false;
    // We may not get this attribute for this node if it does not go through
    // graph rewrite pass. So we do not check for error while retrieving this
    // attribute value.
    OP_REQUIRES_OK(context,
                   context->GetAttr("workspace_enabled", &workspace_enabled_));
  }

  void Compute(OpKernelContext* context) override {
    MklMaxPoolingOpContext mkl_context;
    // Get the input tensor
    const Tensor& tensor_in = MklGetInput(context, 0);
    GetMklShape(context, 0, &mkl_context.input_shape);
    bool input_in_mkl_format = mkl_context.input_shape.IsMklTensor();

    mkl_context.params.in_dim = 4;
    MklPoolParameters pool_params;
    if (input_in_mkl_format == false) {
      pool_params.Init(context, ksize_, stride_, padding_, data_format_,
                       tensor_in.shape());
      OP_REQUIRES(
          context, (pool_params.depth_window == 1),
          errors::Unimplemented("Depthwise max pooling not supported by MKL"));

    } else {
      pool_params.Init(context, ksize_, stride_, padding_, data_format_,
                       &mkl_context.input_shape);
    }

    // Extract the parameters for the op from the pooling specs

    ExtractMklOpParams(context, data_format_, pool_params, &mkl_context.params);

    mkl_context.MklCreateLayoutsAndPrimitives(context);
    OP_REQUIRES_OK(context, context->status());

    // Declare output tensor
    TensorShape tensor_out_shape;
    MklShape mkl_out_shape, mkl_workspace_shape;
    mkl_out_shape.SetMklTensor(true);
    mkl_out_shape.SetMklLayout(mkl_context.prim_pooling_fwd, dnnResourceDst);
    mkl_out_shape.SetTfLayout(mkl_context.params.in_dim,
                              mkl_context.params.out_sizes,
                              mkl_context.params.out_strides);
    mkl_out_shape.SetTfDimOrder(mkl_context.params.in_dim, data_format_);

    Tensor* output_tensor = nullptr;
    tensor_out_shape.AddDim(dnnLayoutGetMemorySize_F32(static_cast<dnnLayout_t>(
                                mkl_out_shape.GetMklLayout())) /
                            sizeof(T));
    AllocateOutputSetMklShape(context, 0, &output_tensor, tensor_out_shape,
                              mkl_out_shape);

    Tensor* workspace_tensor;

    TensorShape workspace_shape;
    mkl_workspace_shape.SetMklTensor(false);
    workspace_shape.AddDim(dnnLayoutGetMemorySize_F32(static_cast<dnnLayout_t>(
                               mkl_context.lt_workspace)) /
                           sizeof(T));
    AllocateOutputSetMklShape(context, 1, &workspace_tensor, workspace_shape,
                              mkl_workspace_shape);

    mkl_context.pooling_res[dnnResourceWorkspace] = const_cast<void*>(
        static_cast<const void*>(workspace_tensor->flat<T>().data()));
    mkl_context.pooling_res[dnnResourceSrc] =
        const_cast<void*>(static_cast<const void*>(tensor_in.flat<T>().data()));
    mkl_context.pooling_res[dnnResourceDst] = const_cast<void*>(
        static_cast<const void*>(output_tensor->flat<T>().data()));

    CHECK_EQ(
        dnnExecute_F32(mkl_context.prim_pooling_fwd, mkl_context.pooling_res),
        E_SUCCESS);

    mkl_context.MklCleanup();
  }

 private:
  typedef struct {
    MklPoolingOpParams params;
    MklShape input_shape;
    void* pooling_res[dnnResourceNumber];
    dnnPrimitive_t prim_pooling_fwd = nullptr;
    dnnLayout_t lt_user_input = nullptr, lt_workspace = nullptr;

    void MklCreateLayoutsAndPrimitives(OpKernelContext* context) {
      bool input_in_mkl_format = input_shape.IsMklTensor();
      // Create or use existing DNN user layout
      if (input_in_mkl_format == false) {
        CHECK_EQ(dnnLayoutCreate_F32(&lt_user_input, params.in_dim,
                                     params.in_sizes, params.in_strides),
                 E_SUCCESS);
      } else {
        lt_user_input = (dnnLayout_t)input_shape.GetCurLayout();
      }

      dnnAlgorithm_t algorithm = dnnAlgorithmPoolingMax;
      dnnPrimitiveAttributes_t primAttr = nullptr;

      // Create DNN primitives
      CHECK_EQ(dnnPoolingCreateForward_F32(
                   &prim_pooling_fwd, primAttr, algorithm, lt_user_input,
                   params.kernel_size, params.kernel_stride, params.in_offset,
                   dnnBorderZerosAsymm),
               E_SUCCESS);

      // Creates layout for the workspace
      CHECK_EQ(dnnLayoutCreateFromPrimitive_F32(&lt_workspace, prim_pooling_fwd,
                                                dnnResourceWorkspace),
               E_SUCCESS);
    }

    void MklCleanup() {
      bool input_in_mkl_format = input_shape.IsMklTensor();
      CHECK_EQ(dnnDelete_F32(prim_pooling_fwd), E_SUCCESS);
      if (!input_in_mkl_format) {
        CHECK_EQ(dnnLayoutDelete_F32(lt_user_input), E_SUCCESS);
      }
      CHECK_EQ(dnnLayoutDelete_F32(lt_workspace), E_SUCCESS);
    }
  } MklMaxPoolingOpContext;

  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;
  bool workspace_enabled_;
};

// The operation to compute MaxPool gradients.
// It takes three inputs:
//   - The original input tensor
//   - The original output tensor
//   - Backprop tensor for output
// It produces one output: backprop tensor for input.
template <class Device, class T>
class MklMaxPoolingGradOp : public OpKernel {
 public:
  explicit MklMaxPoolingGradOp(OpKernelConstruction* context)
      : OpKernel(context) {
    string data_format;

    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
    OP_REQUIRES(context, ksize_.size() == 4,
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 4,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    OP_REQUIRES(context, ksize_[0] == 1 && stride_[0] == 1,
                errors::Unimplemented(
                    "Pooling is not yet supported on the batch dimension."));
    workspace_enabled_ = false;
    // We may not get this attribute for this node if it does not go through
    // graph rewrite pass. So we do not check for error while retrieving this
    // attribute value.
    OP_REQUIRES_OK(context,
                   context->GetAttr("workspace_enabled", &workspace_enabled_));
  }

  void Compute(OpKernelContext* context) override {
    MklMaxPoolingGradOpContext mkl_context;
    // Input - The original input tensor
    const Tensor& tensor_in = MklGetInput(context, 0);

    // Output - Backprop tensor for input.
    Tensor* output_tensor = nullptr;

    GetMklShape(context, 0, &mkl_context.input_shape);
    GetMklShape(context, 2, &mkl_context.output_backprop_shape);
    bool input_in_mkl_format = mkl_context.input_shape.IsMklTensor();

    if (input_in_mkl_format == false)
      mkl_context.params.in_dim = tensor_in.dims();
    else
      mkl_context.params.in_dim = mkl_context.input_shape.GetDimension();

    MklPoolParameters pool_params;
    if (input_in_mkl_format == false) {
      pool_params.Init(context, ksize_, stride_, padding_, data_format_,
                       tensor_in.shape());
      OP_REQUIRES(
          context, (pool_params.depth_window == 1),
          errors::Unimplemented("Depthwise max pooling not supported by MKL"));

    } else {
      pool_params.Init(context, ksize_, stride_, padding_, data_format_,
                       &mkl_context.input_shape);
    }

    // Extract the parameters for the op from the pooling specs
    ExtractMklOpParams(context, data_format_, pool_params, &mkl_context.params);

    mkl_context.MklCreateLayouts(context);
    OP_REQUIRES_OK(context, context->status());

    mkl_context.MklCreatePrimitives(context, workspace_enabled_);
    OP_REQUIRES_OK(context, context->status());

    mkl_context.MklPrepareInputs(context, workspace_enabled_);
    OP_REQUIRES_OK(context, context->status());

    // Create shape for the input back prop output
    TensorShape mkl_input_backprop;
    MklShape mkl_output_shape;
    mkl_output_shape.SetMklTensor(true);
    mkl_output_shape.SetMklLayout(mkl_context.prim_pooling_bwd,
                                  dnnResourceDiffSrc);
    mkl_output_shape.SetTfLayout(mkl_context.params.in_dim,
                                 mkl_context.params.in_sizes,
                                 mkl_context.params.in_strides);
    mkl_output_shape.SetTfDimOrder(mkl_context.params.in_dim, data_format_);

    mkl_input_backprop.AddDim(
        dnnLayoutGetMemorySize_F32(
            static_cast<dnnLayout_t>(mkl_output_shape.GetMklLayout())) /
        sizeof(T));
    AllocateOutputSetMklShape(context, 0, &output_tensor, mkl_input_backprop,
                              mkl_output_shape);
    mkl_context.pooling_res[dnnResourceDiffSrc] = const_cast<void*>(
        static_cast<const void*>(output_tensor->flat<T>().data()));

    CHECK_EQ(
        dnnExecute_F32(mkl_context.prim_pooling_bwd, mkl_context.pooling_res),
        E_SUCCESS);

    mkl_context.MklCleanup(workspace_enabled_);
  }

 private:
  typedef struct {
    MklPoolingOpParams params;
    MklShape input_shape, output_backprop_shape;
    void* pooling_resfwd[dnnResourceNumber];
    void* pooling_res[dnnResourceNumber];
    dnnPrimitive_t prim_pooling_fwd = nullptr, prim_pooling_bwd = nullptr,
                   convert_input = nullptr, convert_outbackprop = nullptr;
    dnnLayout_t lt_outbackprop_user = nullptr, lt_outbackprop_prim = nullptr,
                lt_input_user = nullptr, lt_input_prim = nullptr;
    void* input_buf;
    void* outbackprop_buf;
    Tensor tmp_output_buf_tensor;
    Tensor workspace_buf_tensor;
    Tensor input_buf_tensor, outbackprop_buf_tensor;

    void MklCreateLayouts(OpKernelContext* context) {
      bool input_in_mkl_format = input_shape.IsMklTensor();
      bool outbackprop_in_mkl_format = output_backprop_shape.IsMklTensor();
      // Create DNN user layout for input and outbackprop or get existing layout
      if (input_in_mkl_format == false) {
        CHECK_EQ(dnnLayoutCreate_F32(&lt_input_user, params.in_dim,
                                     params.in_sizes, params.in_strides),
                 E_SUCCESS);
      } else {
        lt_input_user = (dnnLayout_t)input_shape.GetCurLayout();
      }

      // We don't care about the output layout for now as we can create it from
      // primitives for the max pooling fwd prop
      if (outbackprop_in_mkl_format == false) {
        CHECK_EQ(dnnLayoutCreate_F32(&lt_outbackprop_user, params.in_dim,
                                     params.out_sizes, params.out_strides),
                 E_SUCCESS);
      } else {
        lt_outbackprop_user = (dnnLayout_t)output_backprop_shape.GetCurLayout();
      }
    }

    // Create DNN primitives
    void MklCreatePrimitives(OpKernelContext* context, bool workspace_enabled) {
      dnnAlgorithm_t algorithm = dnnAlgorithmPoolingMax;
      dnnPrimitiveAttributes_t primAttr = nullptr;

      if (workspace_enabled == false) {
        CHECK_EQ(dnnPoolingCreateForward_F32(
                     &prim_pooling_fwd, primAttr, algorithm, lt_input_user,
                     params.kernel_size, params.kernel_stride, params.in_offset,
                     dnnBorderZerosAsymm),
                 E_SUCCESS);
      }

      CHECK_EQ(dnnPoolingCreateBackward_F32(
                   &prim_pooling_bwd, primAttr, algorithm, lt_input_user,
                   params.kernel_size, params.kernel_stride, params.in_offset,
                   dnnBorderZerosAsymm),
               E_SUCCESS);

      // Creates conversions
      CHECK_EQ(dnnLayoutCreateFromPrimitive_F32(
                   &lt_outbackprop_prim, prim_pooling_bwd, dnnResourceDiffDst),
               E_SUCCESS);

      if (workspace_enabled == false) {
        CHECK_EQ(dnnLayoutCreateFromPrimitive_F32(
                     &lt_input_prim, prim_pooling_fwd, dnnResourceSrc),
                 E_SUCCESS);
        if (!dnnLayoutCompare_F32(lt_input_user, lt_input_prim)) {
          CHECK_EQ(dnnConversionCreate_F32(&convert_input, lt_input_user,
                                           lt_input_prim),
                   E_SUCCESS);
          AllocTmpBuffer(context, &input_buf_tensor, lt_input_prim, &input_buf);
        }
      }

      if (!dnnLayoutCompare_F32(lt_outbackprop_user, lt_outbackprop_prim)) {
        CHECK_EQ(
            dnnConversionCreate_F32(&convert_outbackprop, lt_outbackprop_user,
                                    lt_outbackprop_prim),
            E_SUCCESS);
        AllocTmpBuffer(context, &outbackprop_buf_tensor, lt_outbackprop_prim,
                       &outbackprop_buf);
      }
    }

    // Compare incoming tensor layouts with MKL preferred layouts and convert
    // data to the preferred layout if necessary
    void MklPrepareInputs(OpKernelContext* context, bool workspace_enabled) {
      const Tensor& tensor_in = MklGetInput(context, 0);
      const Tensor& out_backprop = MklGetInput(context, 2);
      bool input_in_mkl_format = input_shape.IsMklTensor();
      bool outbackprop_in_mkl_format = output_backprop_shape.IsMklTensor();

      void* tmp_output_buf = nullptr;
      void* workspace_buf = nullptr;

      if (workspace_enabled == false) {
        if (convert_input != nullptr) {
          if (input_in_mkl_format == false) {
            CHECK_EQ(dnnConversionExecute_F32(
                         convert_input,
                         const_cast<void*>(static_cast<const void*>(
                             tensor_in.flat<T>().data())),
                         input_buf),
                     E_SUCCESS);
            CHECK_EQ(dnnDelete_F32(convert_input), E_SUCCESS);
            convert_input = nullptr;
          } else {
            input_shape.GetConvertedFlatData(
                lt_input_prim,
                const_cast<void*>(
                    static_cast<const void*>(tensor_in.flat<T>().data())),
                input_buf);
          }
          pooling_resfwd[dnnResourceSrc] = input_buf;
        } else {
          pooling_resfwd[dnnResourceSrc] = const_cast<void*>(
              static_cast<const void*>(tensor_in.flat<T>().data()));
        }

        dnnLayout_t lt_workspace;
        CHECK_EQ(dnnLayoutCreateFromPrimitive_F32(
                     &lt_workspace, prim_pooling_fwd, dnnResourceWorkspace),
                 E_SUCCESS);
        AllocTmpBuffer(context, &workspace_buf_tensor, lt_workspace,
                       &workspace_buf);
        pooling_resfwd[dnnResourceWorkspace] = workspace_buf;

        dnnLayoutDelete_F32(lt_workspace);

        // We create the layout for max pooling fwd prop tmp output here
        AllocTmpBuffer(context, &tmp_output_buf_tensor, lt_outbackprop_prim,
                       &tmp_output_buf);
        pooling_resfwd[dnnResourceDst] = tmp_output_buf;

        CHECK_EQ(dnnExecute_F32(prim_pooling_fwd, pooling_resfwd), E_SUCCESS);
        pooling_res[dnnResourceWorkspace] =
            pooling_resfwd[dnnResourceWorkspace];
      } else {
        const Tensor& workspace = MklGetInput(context, 3);
        pooling_res[dnnResourceWorkspace] = const_cast<void*>(
            static_cast<const void*>(workspace.flat<T>().data()));
      }

      // Out backprop conversions if needed
      if (convert_outbackprop != nullptr) {
        if (outbackprop_in_mkl_format == false) {
          CHECK_EQ(dnnConversionExecute_F32(
                       convert_outbackprop,
                       const_cast<void*>(static_cast<const void*>(
                           out_backprop.flat<T>().data())),
                       outbackprop_buf),
                   E_SUCCESS);
          CHECK_EQ(dnnDelete_F32(convert_outbackprop), E_SUCCESS);
        } else {
          output_backprop_shape.GetConvertedFlatData(
              lt_outbackprop_prim,
              const_cast<void*>(
                  static_cast<const void*>(out_backprop.flat<T>().data())),
              outbackprop_buf);
        }
        pooling_res[dnnResourceDiffDst] = outbackprop_buf;
      } else {
        pooling_res[dnnResourceDiffDst] = const_cast<void*>(
            static_cast<const void*>(out_backprop.flat<T>().data()));
      }
    }

    void MklCleanup(bool workspace_enabled) {
      bool input_in_mkl_format = input_shape.IsMklTensor();
      bool outbackprop_in_mkl_format = output_backprop_shape.IsMklTensor();
      if (workspace_enabled == false) {
        CHECK_EQ(dnnDelete_F32(prim_pooling_fwd), E_SUCCESS);
      }
      CHECK_EQ(dnnDelete_F32(prim_pooling_bwd), E_SUCCESS);
      if (outbackprop_in_mkl_format == false) {
        CHECK_EQ(dnnLayoutDelete_F32(lt_outbackprop_user), E_SUCCESS);
      }
      CHECK_EQ(dnnLayoutDelete_F32(lt_outbackprop_prim), E_SUCCESS);
      if (input_in_mkl_format == false) {
        CHECK_EQ(dnnLayoutDelete_F32(lt_input_user), E_SUCCESS);
      }
      if (workspace_enabled == false) {
        CHECK_EQ(dnnLayoutDelete_F32(lt_input_prim), E_SUCCESS);
      }
    }
  } MklMaxPoolingGradOpContext;

  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;

  bool workspace_enabled_;
};  // MklMaxPoolingGradOp

#else

// An implementation of MaxPooling (forward).
template <typename Device, typename T>
class MklMaxPoolingOp : public MklPoolingForwardOpBase<T> {
 public:
  explicit MklMaxPoolingOp(OpKernelConstruction* context)
      : MklPoolingForwardOpBase<T>(context) {
    // In Max Pooling, MKLDNN does not allow passing workspace as NULL.
    // So we set workspace_enabled_ to true.
    this->workspace_enabled_ = true;
  }

  void Compute(OpKernelContext* context) override {
    try {
      auto cpu_engine = engine(engine::cpu, 0);
      const Tensor& input_tensor =
          MklGetInput(context, this->kInputTensorIndexInput);
      MklDnnShape dnn_shape_input;
      GetMklShape(context, this->kInputTensorIndexInput, &dnn_shape_input);
      this->SanityCheckInput(context, input_tensor, dnn_shape_input);
      if (!context->status().ok()) return;

      MklDnnData<T> dnn_data_input(&cpu_engine);
      MklDnnData<T> dnn_data_output(&cpu_engine);
      MklDnnData<uint8> dnn_data_wksp(&cpu_engine);

      // initialize variables for the pooling op
      MklPoolParameters pool_params;
      // Get the input tensor and initialize the pooling parameters
      this->ConfigureInput(context, dnn_shape_input, input_tensor, &pool_params,
                           &dnn_data_input);
      OP_REQUIRES_OK(context, context->status());

      // Declare output tensor
      Tensor* output_tensor = nullptr;
      memory::dims output_dims_mkl_order;
      this->GetOutputDims(pool_params, &output_dims_mkl_order);

      // If input is in Mkl layout, then just get the memory format from it
      // directly, instead of using input data_format to MaxPool.
      if (dnn_shape_input.IsMklTensor()) {
        dnn_data_output.SetUsrMem(
            output_dims_mkl_order,
            static_cast<memory::format>(
                dnn_data_input.GetUsrMemDesc().data.format));
      } else {
        dnn_data_output.SetUsrMem(output_dims_mkl_order,
                                  this->data_format_mkldnn_);
      }

      // describe the memory layout; let mkl-dnn choose the best for the op
      dnn_data_output.SetOpMemDesc(output_dims_mkl_order, memory::format::any);

      auto pool_desc = pooling_forward::desc(
          prop_kind::forward, algorithm::pooling_max,
          dnn_data_input.GetUsrMemDesc(), dnn_data_output.GetUsrMemDesc(),
          memory::dims({pool_params.row_stride, pool_params.col_stride}),
          memory::dims({pool_params.window_rows, pool_params.window_cols}),
          memory::dims({static_cast<int>(pool_params.pad_top),
                        static_cast<int>(pool_params.pad_left)}),
          memory::dims({static_cast<int>(pool_params.pad_bottom),
                        static_cast<int>(pool_params.pad_right)}),
          TFPaddingToMklDnnPadding(this->padding_));
      auto pool_fwd_desc =
          pooling_forward::primitive_desc(pool_desc, cpu_engine);

      this->AllocateOutputTensor(context, pool_fwd_desc, output_dims_mkl_order,
                                 this->data_format_mkldnn_, &output_tensor);
      OP_REQUIRES_OK(context, context->status());
      dnn_data_output.SetUsrMemDataHandle(output_tensor);

      AllocateWorkspaceTensor(context, pool_fwd_desc, &dnn_data_wksp);
      OP_REQUIRES_OK(context, context->status());

      this->PrepareAndExecuteNet(pool_fwd_desc, &dnn_data_input,
                                 &dnn_data_output, &dnn_data_wksp);
    } catch (mkldnn::error& e) {
      string error_msg = "Status: " + std::to_string(e.status) +
                         ", message: " + string(e.message) + ", in file " +
                         string(__FILE__) + ":" + std::to_string(__LINE__);
      OP_REQUIRES_OK(context, errors::Aborted("Compute received an exception:",
                                              error_msg));
    }
  }  // Compute

 private:
  const int kOutputTensorIndexWorkspace = 1;

  void AllocateWorkspaceTensor(
      OpKernelContext* context,
      const pooling_forward::primitive_desc& pool_fwd_prim_desc,
      MklDnnData<uint8>* dnn_data_wksp) {
    CHECK_NOTNULL(dnn_data_wksp);
    Tensor* workspace_tensor = nullptr;
    memory::primitive_desc workspace_pd =
        pool_fwd_prim_desc.workspace_primitive_desc();
    size_t workspace_bytes = workspace_pd.get_size();
    MklDnnShape workspace_mkl_shape;
    workspace_mkl_shape.SetMklTensor(false);
    TensorShape workspace_tf_shape;
    workspace_tf_shape.AddDim(workspace_bytes);
    AllocateOutputSetMklShape(context, kOutputTensorIndexWorkspace,
                              &workspace_tensor, workspace_tf_shape,
                              workspace_mkl_shape);
    CHECK_NOTNULL(workspace_tensor);
    dnn_data_wksp->SetUsrMem(workspace_pd, workspace_tensor);
  }
};

// The operation to compute MaxPool gradients.
// It takes three inputs:
//   - The original input tensor
//   - The original output tensor
//   - Backprop tensor for output
// It produces one output: backprop tensor for input.
template <class Device, class T>
class MklMaxPoolingGradOp : public MklPoolingBackwardOpBase<T> {
 public:
  explicit MklMaxPoolingGradOp(OpKernelConstruction* context)
      : MklPoolingBackwardOpBase<T>(context) {}

  void Compute(OpKernelContext* context) override {
    try {
      auto cpu_engine = engine(engine::cpu, 0);
      const Tensor& orig_input_tensor =
          MklGetInput(context, kInputTensorIndexOrigInput);
      const Tensor& orig_output_tensor =
          MklGetInput(context, kInputTensorIndexOrigOutput);
      const Tensor& grad_tensor =
          MklGetInput(context, kInputTensorIndexGradient);
      const Tensor& workspace_tensor =
          MklGetInput(context, kInputTensorIndexWorkspace);
      MklDnnShape orig_input_mkl_shape, orig_output_mkl_shape, grad_mkl_shape,
          workspace_mkl_shape;
      GetMklShape(context, kInputTensorIndexOrigInput, &orig_input_mkl_shape);
      GetMklShape(context, kInputTensorIndexOrigOutput, &orig_output_mkl_shape);
      GetMklShape(context, kInputTensorIndexGradient, &grad_mkl_shape);
      GetMklShape(context, kInputTensorIndexWorkspace, &workspace_mkl_shape);

      SanityCheckInputs(context, orig_input_tensor, orig_output_tensor,
                        grad_tensor, workspace_tensor, orig_input_mkl_shape,
                        orig_output_mkl_shape, grad_mkl_shape,
                        workspace_mkl_shape);
      if (!context->status().ok()) return;

      MklDnnData<T> grad_dnn_data(&cpu_engine);
      MklDnnData<uint8> workspace_dnn_data(&cpu_engine);
      MklDnnData<T> output_dnn_data(&cpu_engine);
      Tensor* output_tensor = nullptr;
      MklPoolParameters pool_params;
      TensorShape orig_input_shape;
      memory::dims output_dims_mkl_order, orig_input_dims_mkl_order;
      memory::desc original_input_md = ConfigureOriginalInput(
          context, orig_input_tensor, orig_input_mkl_shape,
          &orig_input_dims_mkl_order, &pool_params, &orig_input_shape);

      memory::desc original_output_md = this->ConfigureOriginalOutput(
          pool_params, orig_output_mkl_shape, output_dims_mkl_order);

      memory::desc target_diff_dst_md = this->ConfigureInputGradient(
          grad_mkl_shape, grad_tensor, &grad_dnn_data, original_output_md);

      output_dnn_data.SetUsrMem(original_input_md);

      // Create the forward pooling primitive descriptor so we can
      // pass it as a hint to the backward pooling primitive descriptor
      auto pool_fwd_desc = pooling_forward::desc(
          prop_kind::forward, algorithm::pooling_max, original_input_md,
          original_output_md,
          memory::dims({pool_params.row_stride, pool_params.col_stride}),
          memory::dims({pool_params.window_rows, pool_params.window_cols}),
          memory::dims({static_cast<int>(pool_params.pad_top),
                        static_cast<int>(pool_params.pad_left)}),
          memory::dims({static_cast<int>(pool_params.pad_bottom),
                        static_cast<int>(pool_params.pad_right)}),
          TFPaddingToMklDnnPadding(this->padding_));
      auto pool_fwd_prim_desc =
          pooling_forward::primitive_desc(pool_fwd_desc, cpu_engine);

      auto pool_bkwd_desc = pooling_backward::desc(
          algorithm::pooling_max, output_dnn_data.GetUsrMemDesc(),
          target_diff_dst_md,
          memory::dims({pool_params.row_stride, pool_params.col_stride}),
          memory::dims({pool_params.window_rows, pool_params.window_cols}),
          memory::dims({static_cast<int>(pool_params.pad_top),
                        static_cast<int>(pool_params.pad_left)}),
          memory::dims({static_cast<int>(pool_params.pad_bottom),
                        static_cast<int>(pool_params.pad_right)}),
          TFPaddingToMklDnnPadding(this->padding_));
      auto pool_bkwd_prim_desc = pooling_backward::primitive_desc(
          pool_bkwd_desc, cpu_engine, pool_fwd_prim_desc);

      this->AllocateOutputTensor(context, pool_bkwd_prim_desc,
                                 orig_input_dims_mkl_order,
                                 this->data_format_mkldnn_, &output_tensor);
      output_dnn_data.SetUsrMemDataHandle(output_tensor);

      ConfigureWorkspace(workspace_tensor,
                         pool_fwd_prim_desc.workspace_primitive_desc(),
                         &workspace_dnn_data);
      this->PrepareAndExecuteNet(
          pool_bkwd_prim_desc, &grad_dnn_data, &output_dnn_data,
          memory::primitive_desc(target_diff_dst_md, cpu_engine),
          &workspace_dnn_data);
    } catch (mkldnn::error& e) {
      string error_msg = "Status: " + std::to_string(e.status) +
                         ", message: " + string(e.message) + ", in file " +
                         string(__FILE__) + ":" + std::to_string(__LINE__);
      OP_REQUIRES_OK(context, errors::Aborted("Compute received an exception:",
                                              error_msg));
    }
  }  // Compute

 private:
  // .Input("orig_input: T")
  // .Input("orig_output: T")
  // .Input("grad: T")
  // .Input("workspace: T")
  const int kInputTensorIndexOrigInput = 0;
  const int kInputTensorIndexOrigOutput = 1;
  const int kInputTensorIndexGradient = 2;
  const int kInputTensorIndexWorkspace = 3;
  //  Output("output: T") in Base Class

  memory::desc ConfigureOriginalInput(
      OpKernelContext* context, const Tensor& tensor_original_input,
      const MklDnnShape& original_input_mkl_shape,
      memory::dims* original_input_dims_mkl_order,
      MklPoolParameters* pool_params, TensorShape* input_tensor_shape) {
    *input_tensor_shape = tensor_original_input.shape();
    return MklPoolingBackwardOpBase<T>::ConfigureOriginalInput(
        context, tensor_original_input, original_input_mkl_shape,
        original_input_dims_mkl_order, pool_params, *input_tensor_shape);
  }

  void ConfigureWorkspace(const Tensor& workspace_tensor,
                          memory::primitive_desc workspace_pd,
                          MklDnnData<uint8>* workspace_dnn_data) {
    CHECK_NOTNULL(workspace_dnn_data);

    workspace_dnn_data->SetUsrMem(workspace_pd, &workspace_tensor);
  }

  void SanityCheckInputs(OpKernelContext* context,
                         const Tensor& orig_input_tensor,
                         const Tensor& orig_output_tensor,
                         const Tensor& grad_tensor,
                         const Tensor& workspace_tensor,
                         const MklDnnShape& orig_input_mkl_shape,
                         const MklDnnShape& orig_output_mkl_shape,
                         const MklDnnShape& grad_mkl_shape,
                         const MklDnnShape& workspace_mkl_shape) {
    if (!orig_input_mkl_shape.IsMklTensor()) {
      OP_REQUIRES(context, orig_input_tensor.dims() == 4,
                  errors::InvalidArgument("Original input shape must be "
                                          "4-dimensional"));
    } else {
      OP_REQUIRES(context, orig_input_mkl_shape.GetDimension() == 4,
                  errors::InvalidArgument("Original input shape must be "
                                          "4-dimensional"));
    }
    if (!orig_output_mkl_shape.IsMklTensor()) {
      OP_REQUIRES(context, orig_output_tensor.dims() == 4,
                  errors::InvalidArgument("Original output must be "
                                          "4-dimensional"));
    } else {
      OP_REQUIRES(context, orig_output_mkl_shape.GetDimension() == 4,
                  errors::InvalidArgument("Original output must be "
                                          "4-dimensional"));
    }
    if (!grad_mkl_shape.IsMklTensor()) {
      OP_REQUIRES(context, grad_tensor.dims() == 4,
                  errors::InvalidArgument("Gradient must be 4-dimensional"));
    } else {
      OP_REQUIRES(context, grad_mkl_shape.GetDimension() == 4,
                  errors::InvalidArgument("Gradient must be "
                                          "4-dimensional"));
    }
    if (this->workspace_enabled_) {
      // The workspace should not be an MKL tensor
      OP_REQUIRES(context, workspace_mkl_shape.IsMklTensor() == false,
                  errors::InvalidArgument("Workspace tensor should not"
                                          " be an MKL Tensor."));
      // It should only have one dimension
      OP_REQUIRES(context, workspace_tensor.dims() == 1,
                  errors::InvalidArgument("Workspace tensor must be "
                                          "1-dimensional"));
    } else {
      OP_REQUIRES(
          context, this->workspace_enabled_,
          errors::Unimplemented("MKL-DNN Max Pooling does not "
                                "yet support the use case "
                                "where MaxPoolGrad is called without first"
                                " calling MaxPool."));
    }
  }
};  // MklMaxPoolingGradOp

#endif  // INTEL_MKL_ML

REGISTER_KERNEL_BUILDER(Name("_MklMaxPool")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<float>("T")
                            .Label(mkl_op_registry::kMklOpLabel),
                        MklMaxPoolingOp<CPUDevice, float>);

REGISTER_KERNEL_BUILDER(Name("_MklMaxPoolGrad")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<float>("T")
                            .Label(mkl_op_registry::kMklOpLabel),
                        MklMaxPoolingGradOp<CPUDevice, float>);

}  // namespace tensorflow
#endif  // INTEL_MKL
