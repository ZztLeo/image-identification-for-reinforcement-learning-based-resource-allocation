/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#define EIGEN_USE_THREADS
#include "tensorflow/core/kernels/variable_ops.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// Resource stored by variables in the resource manager
// (legacy, ref-style version).
class LegacyVar : public ResourceBase {
 public:
  explicit LegacyVar(DataType dtype) : tensor_(dtype) {}
  // Not copyable or movable.
  LegacyVar(const LegacyVar&) = delete;
  LegacyVar& operator=(const LegacyVar&) = delete;

  mutex* mu() { return &mu_; }
  Tensor* tensor() { return &tensor_; }

  string DebugString() override {
    return strings::StrCat(DataTypeString(tensor_.dtype()), "/",
                           tensor_.shape().DebugString());
  }

 private:
  mutex mu_;
  Tensor tensor_;

  ~LegacyVar() override {}
};

VariableOp::VariableOp(OpKernelConstruction* context) : OpKernel(context) {
  OP_REQUIRES_OK(context, context->GetAttr("shape", &shape_));
  dtype_ = RemoveRefType(context->output_type(0));
}

void VariableOp::Compute(OpKernelContext* ctx) {
  mutex_lock l(init_mu_);
  if (!initialized_) {
    OP_REQUIRES_OK(ctx, cinfo_.Init(ctx->resource_manager(), def(),
                                    true /* use name() */));
    initialized_ = true;
  }
  auto creator = [this](LegacyVar** var) {
    *var = new LegacyVar(dtype_);
    (*var)->tensor()->set_shape(shape_);
    return Status::OK();
  };
  LegacyVar* var;
  OP_REQUIRES_OK(ctx, cinfo_.resource_manager()->LookupOrCreate<LegacyVar>(
                          cinfo_.container(), cinfo_.name(), &var, creator));
  // Output a reference to our tensor, so it may be updated.
  //
  // As long as the resource manager hasn't been cleared the ref we return
  // here is valid because it owns a ref on var.
  ctx->set_output_ref(0, var->mu(), var->tensor());
  if (ctx->track_allocations() && var->tensor()->IsInitialized()) {
    AllocatorAttributes attr;
    attr.set_gpu_compatible(true);
    attr.set_nic_compatible(true);
    ctx->record_persistent_memory_allocation(var->tensor()->AllocatedBytes());
  }
  var->Unref();
}

class TemporaryVariableOp : public OpKernel {
 public:
  explicit TemporaryVariableOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("shape", &shape_));
    OP_REQUIRES_OK(context, context->GetAttr("dtype", &dtype_));
    OP_REQUIRES_OK(context, context->GetAttr("var_name", &var_name_));
    // Variable name defaults to op name if not specified explicitly.
    if (var_name_.empty()) var_name_ = name();
  }

  void Compute(OpKernelContext* context) override {
    Status s;
    ResourceMgr* rm = context->resource_manager();
    OP_REQUIRES(context, rm, errors::Internal("No per-step resource manager."));
    auto* tmp_var = new TmpVar;
    OP_REQUIRES(context, tmp_var,
                errors::ResourceExhausted("Could not allocate TmpVar."));
    tmp_var->name = var_name_;
    s = context->allocate_temp(dtype_, shape_, &tmp_var->val);
    if (!s.ok()) tmp_var->Unref();
    OP_REQUIRES_OK(context, s);
    OP_REQUIRES_OK(context, rm->Create(context->step_container()->name(),
                                       var_name_, tmp_var));
    context->set_output_ref(0, &tmp_var->mu, &tmp_var->val);
    if (context->track_allocations()) {
      context->record_persistent_memory_allocation(
          tmp_var->val.AllocatedBytes());
    }
  }

 private:
  // Refcounted temporary variable resource.
  friend class DestroyTemporaryVariableOp;
  struct TmpVar : public ResourceBase {
    mutex mu;
    Tensor val;
    string name;
    string DebugString() override { return name; }
    ~TmpVar() override { VLOG(3) << "TmpVar " << name << " deleted"; }
  };

  TensorShape shape_;
  DataType dtype_;
  string var_name_;
};

class DestroyTemporaryVariableOp : public OpKernel {
 public:
  explicit DestroyTemporaryVariableOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES(context, IsRefType(context->input_type(0)),
                errors::InvalidArgument("lhs input needs to be a ref type"));
    OP_REQUIRES_OK(context, context->GetAttr("var_name", &var_name_));
    OP_REQUIRES(context, !var_name_.empty(),
                errors::InvalidArgument("Missing var_name attribute"));
  }

  void Compute(OpKernelContext* context) override {
    // NOTE(pbar): All other mutators of the Tensor Ref *must* have completed
    // their execution before this DestroyTemporaryVariable op executes.
    // This is typically achieved using control dependencies.
    CHECK(IsRefType(context->input_dtype(0)));
    Tensor tmpvar = context->mutable_input(0, false);
    context->set_output(0, tmpvar);
    ResourceMgr* rm = context->resource_manager();
    OP_REQUIRES(context, rm, errors::Internal("No per-step resource manager."));
    OP_REQUIRES_OK(context, rm->Delete<TemporaryVariableOp::TmpVar>(
                                context->step_container()->name(), var_name_));
    if (context->track_allocations()) {
      context->record_persistent_memory_allocation(
          -static_cast<int64>(tmpvar.AllocatedBytes()));
    }
  }

 private:
  string var_name_;
};

class IsVariableInitializedOp : public OpKernel {
 public:
  explicit IsVariableInitializedOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Get a mutable input tensor of the Ref input.
    const Tensor& input_tensor = context->mutable_input(0, false);
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({}), &output));
    auto output_tensor = output->tensor<bool, 0>();
    bool result = input_tensor.IsInitialized();
    output_tensor() = result;
  }
};

REGISTER_KERNEL_BUILDER(Name("Variable").Device(DEVICE_CPU), VariableOp);
REGISTER_KERNEL_BUILDER(Name("VariableV2").Device(DEVICE_CPU), VariableOp);
REGISTER_KERNEL_BUILDER(Name("TemporaryVariable").Device(DEVICE_CPU),
                        TemporaryVariableOp);
REGISTER_KERNEL_BUILDER(Name("DestroyTemporaryVariable").Device(DEVICE_CPU),
                        DestroyTemporaryVariableOp);
REGISTER_KERNEL_BUILDER(Name("IsVariableInitialized").Device(DEVICE_CPU),
                        IsVariableInitializedOp);

#ifdef TENSORFLOW_USE_SYCL
#define REGISTER_SYCL_KERNEL(type)                                          \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("Variable").Device(DEVICE_SYCL).TypeConstraint<type>("dtype"),   \
      VariableOp);                                                          \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("VariableV2").Device(DEVICE_SYCL).TypeConstraint<type>("dtype"), \
      VariableOp);                                                          \
  REGISTER_KERNEL_BUILDER(Name("TemporaryVariable")                         \
                              .Device(DEVICE_SYCL)                          \
                              .TypeConstraint<type>("dtype"),               \
                          TemporaryVariableOp);                             \
  REGISTER_KERNEL_BUILDER(Name("DestroyTemporaryVariable")                  \
                              .Device(DEVICE_SYCL)                          \
                              .TypeConstraint<type>("T"),                   \
                          DestroyTemporaryVariableOp);                      \
  REGISTER_KERNEL_BUILDER(Name("IsVariableInitialized")                     \
                              .Device(DEVICE_SYCL)                          \
                              .TypeConstraint<type>("dtype")                \
                              .HostMemory("is_initialized"),                \
                          IsVariableInitializedOp);

TF_CALL_GPU_NUMBER_TYPES_NO_HALF(REGISTER_SYCL_KERNEL);
#undef REGISTER_SYCL_KERNEL
#endif  // TENSORFLOW_USE_SYCL

#if GOOGLE_CUDA
// Only register 'Variable' on GPU for the subset of types also supported by
// 'Assign' (see dense_update_ops.cc.)
#define REGISTER_GPU_KERNELS(type)                                         \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("Variable").Device(DEVICE_GPU).TypeConstraint<type>("dtype"),   \
      VariableOp);                                                         \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("VariableV2").Device(DEVICE_GPU).TypeConstraint<type>("dtype"), \
      VariableOp);                                                         \
  REGISTER_KERNEL_BUILDER(Name("TemporaryVariable")                        \
                              .Device(DEVICE_GPU)                          \
                              .TypeConstraint<type>("dtype"),              \
                          TemporaryVariableOp);                            \
  REGISTER_KERNEL_BUILDER(Name("DestroyTemporaryVariable")                 \
                              .Device(DEVICE_GPU)                          \
                              .TypeConstraint<type>("T"),                  \
                          DestroyTemporaryVariableOp);                     \
  REGISTER_KERNEL_BUILDER(Name("IsVariableInitialized")                    \
                              .Device(DEVICE_GPU)                          \
                              .TypeConstraint<type>("dtype")               \
                              .HostMemory("is_initialized"),               \
                          IsVariableInitializedOp);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNELS);
TF_CALL_int64(REGISTER_GPU_KERNELS);
#undef REGISTER_GPU_KERNELS
#endif  // GOOGLE_CUDA

}  // namespace tensorflow
