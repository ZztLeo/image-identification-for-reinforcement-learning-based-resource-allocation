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
// This file defines a C API for implementing operations in tflite.
// These operations can be defined using c++ but the interface between
// the interpreter and the operations are C.
//
// Summary of abstractions
// TF_LITE_ENSURE - Self-sufficient error checking
// TfLiteStatus - Status reporting
// TfLiteIntArray - stores tensor shapes (dims),
// TfLiteContext - allows an op to access the tensors
// TfLiteTensor - tensor (a multidimensional array)
// TfLiteNode - a single node or operation
// TfLiteRegistration - the implementation of a conceptual operation.
//
// Some abstractions in this file are created and managed by Interpreter.
#ifndef TENSORFLOW_CONTRIB_LITE_CONTEXT_H_
#define TENSORFLOW_CONTRIB_LITE_CONTEXT_H_

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef enum { kTfLiteOk = 0, kTfLiteError = 1 } TfLiteStatus;

// Forward declare so GetNode can use this is in Context.
typedef struct _TfLiteRegistration TfLiteRegistration;
typedef struct _TfLiteDelegate TfLiteDelegate;

#define kOptionalTensor (-1)

// Fixed size list of integers. Used for dimensions and inputs/outputs tensor
// indices
typedef struct {
  int size;
// gcc 6.1+ have a bug where flexible members aren't properly handled
// https://github.com/google/re2/commit/b94b7cd42e9f02673cd748c1ac1d16db4052514c
#if !defined(__clang__) && defined(__GNUC__) && __GNUC__ == 6 && \
    __GNUC_MINOR__ >= 1
  int data[0];
#else
  int data[];
#endif
} TfLiteIntArray;

// Given the size (number of elements) in a TfLiteIntArray, calculate its size
// in bytes.
int TfLiteIntArrayGetSizeInBytes(int size);

// Create a array of a given `size` (uninitialized entries).
// This returns a pointer, that you must free using TfLiteIntArrayFree().
TfLiteIntArray* TfLiteIntArrayCreate(int size);

// Check if two tensors are equal. Returns 1 if they are equal, 0 otherwise.
int TfLiteIntArrayEqual(TfLiteIntArray* a, TfLiteIntArray* b);

// Create a copy of an array passed as `src`.
// You are expected to free memory with TfLiteIntArrayFree
TfLiteIntArray* TfLiteIntArrayCopy(TfLiteIntArray* src);

// Free memory of array `v`.
void TfLiteIntArrayFree(TfLiteIntArray* v);

// Since we must not depend on any libraries, define a minimal subset of
// error macros while avoiding names that have pre-conceived meanings like
// assert and check.

// Check whether value is true, and if not return kTfLiteError from
// the current function (and report the error string msg).
#define TF_LITE_ENSURE_MSG(context, value, msg)            \
  do {                                                     \
    if (!(value)) {                                        \
      (context)->ReportError((context), __FILE__ " " msg); \
      return kTfLiteError;                                 \
    }                                                      \
  } while (0)

// Check whether the value `a` is true, and if not return kTfLiteError from
// the current function, while also reporting the location of the error.
#define TF_LITE_ENSURE(context, a)                                          \
  do {                                                                      \
    if (!(a)) {                                                             \
      (context)->ReportError((context), "%s:%d %s was not true.", __FILE__, \
                             __LINE__, #a);                                 \
      return kTfLiteError;                                                  \
    }                                                                       \
  } while (0)

#define TF_LITE_ENSURE_STATUS(a) \
  do {                           \
    if ((a) != kTfLiteOk) {      \
      return kTfLiteError;       \
    }                            \
  } while (0)

// Check whether the value `a == b` is true, and if not return kTfLiteError from
// the current function, while also reporting the location of the error.
// `a` and `b` may be evaluated more than once, so no side effects or
// extremely expensive computations should be done.
#define TF_LITE_ENSURE_EQ(context, a, b)                                       \
  do {                                                                         \
    if ((a) != (b)) {                                                          \
      (context)->ReportError((context), "%s:%d %s != %s (%d != %d)", __FILE__, \
                             __LINE__, #a, #b, (a), (b));                      \
      return kTfLiteError;                                                     \
    }                                                                          \
  } while (0)

#define TF_LITE_ENSURE_OK(context, status) \
  do {                                     \
    if ((status) != kTfLiteOk) {           \
      return status;                       \
    }                                      \
  } while (0)

// Types supported by tensor
typedef enum {
  kTfLiteNoType = 0,
  kTfLiteFloat32 = 1,
  kTfLiteInt32 = 2,
  kTfLiteUInt8 = 3,
  kTfLiteInt64 = 4,
  kTfLiteString = 5,
} TfLiteType;

// Parameters for asymmetric quantization. Quantized values can be converted
// back to float using:
//    real_value = scale * (quantized_value - zero_point);
typedef struct {
  float scale;
  int32_t zero_point;
} TfLiteQuantizationParams;

// A union of points that points to memory for a given tensor.
typedef union {
  int* i32;
  int64_t* i64;
  float* f;
  char* raw;
  const char* raw_const;
  uint8_t* uint8;
} TfLitePtrUnion;

// Memory allocation strategies. kTfLiteMmapRo is for read-only memory-mapped
// data (or data externally allocated). kTfLiteArenaRw is arena allocated
// data. kTfLiteDynamic is for tensors that are allocated during evaluation.
typedef enum {
  kTfLiteMemNone = 0,
  kTfLiteMmapRo,
  kTfLiteArenaRw,
  kTfLiteArenaRwPersistent,
  kTfLiteDynamic,
} TfLiteAllocationType;

// The delegates should use zero or positive integers to represent handles.
// -1 is reserved from unallocated status.
typedef int TfLiteBufferHandle;
const TfLiteBufferHandle kTfLiteNullBufferHandle = -1;

// An tensor in the interpreter system which is a wrapper around a buffer of
// data including a dimensionality (or NULL if not currently defined).
typedef struct {
  // The data type specification for data stored in `data`. This affects
  // what member of `data` union should be used.
  TfLiteType type;
  // A union of data pointers. The appropriate type should be used for a typed
  // tensor based on `type`.
  TfLitePtrUnion data;
  // A pointer to a structure representing the dimensionality interpretation
  // that the buffer should have. NOTE: the product of elements of `dims`
  // and the element datatype size should be equal to `bytes` below.
  TfLiteIntArray* dims;
  // Quantization information.
  TfLiteQuantizationParams params;
  // How memory is mapped
  //  kTfLiteMmapRo: Memory mapped read only.
  //  i.e. weights
  //  kTfLiteArenaRw: Arena allocated read write memory
  //  (i.e. temporaries, outputs).
  TfLiteAllocationType allocation_type;
  // The number of bytes required to store the data of this Tensor. I.e.
  // (bytes of each element) * dims[0] * ... * dims[n-1].  For example, if
  // type is kTfLiteFloat32 and dims = {3, 2} then
  // bytes = sizeof(float) * 3 * 2 = 4 * 3 * 2 = 24.
  size_t bytes;

  // An opaque pointer to a tflite::MMapAllocation
  const void* allocation;

  // Null-terminated name of this tensor.
  const char* name;

  // The delegate which knows how to handle `buffer_handle`.
  // WARNING: This is an experimental interface that is subject to change.
  TfLiteDelegate* delegate;

  // An integer buffer handle that can be handled by `delegate`.
  // The value is valid only when delegate is not null.
  // WARNING: This is an experimental interface that is subject to change.
  TfLiteBufferHandle buffer_handle;

  // If the delegate uses its own buffer (e.g. GPU memory), the delegate is
  // responsible to set data_is_stale to true.
  // `delegate->CopyFromBufferHandle` can be called to copy the data from
  // delegate buffer.
  // WARNING: This is an // experimental interface that is subject to change.
  bool data_is_stale;
} TfLiteTensor;

// Free data memory of tensor `t`;
void TfLiteTensorDataFree(TfLiteTensor* t);

// Free memory of tensor `t`;
void TfLiteTensorFree(TfLiteTensor* t);

// Set all of a tensor's fields (and free any previously allocated data).
void TfLiteTensorReset(TfLiteType type, const char* name, TfLiteIntArray* dims,
                       TfLiteQuantizationParams quantization, char* buffer,
                       size_t size, TfLiteAllocationType allocation_type,
                       const void* allocation, TfLiteTensor* tensor);

// Resize the allocated data of a (dynamic) tensor.
void TfLiteTensorRealloc(size_t num_bytes, TfLiteTensor* tensor);

// A structure representing an instance of a node.
// This structure only exhibits the inputs, outputs and user defined data, not
// other features like the type.
typedef struct {
  // Inputs to this node expressed as indices into the simulator's tensors.
  TfLiteIntArray* inputs;

  // Outputs to this node expressed as indices into the simulator's tensors.
  TfLiteIntArray* outputs;

  // Temporary tensors uses during the computations. This usually contains no
  // tensors, but ops are allowed to change that if they need scratch space of
  // any sort.
  TfLiteIntArray* temporaries;

  // Opaque data provided by the node implementer through `Registration.init`.
  void* user_data;

  // Opaque data provided to the node if the node is a builtin. This is usually
  // a structure defined in builtin_op_data.h
  void* builtin_data;

  // Custom initial data. This is the opaque data provided in the flatbuffer.
  // WARNING: This is an experimental interface that is subject to change.
  const void* custom_initial_data;
  int custom_initial_data_size;

  // The pointer to the delegate. This is non-null only when the node is
  // created by calling `interpreter.ModifyGraphWithDelegate`.
  // WARNING: This is an experimental interface that is subject to change.
  TfLiteDelegate* delegate;
} TfLiteNode;

typedef struct TfLiteContext {
  // Number of tensors in the context.
  int tensors_size;

  // The execution plan contains a list of the node indices in execution
  // order. execution_plan->size is the current number of nodes. And,
  // execution_plan->data[0] is the first node that needs to be run.
  // TfLiteDelegates can traverse the current execution plan by iterating
  // through each member of this array and using GetNodeAndRegistration() to
  // access details about a node. i.e.
  // TfLiteIntArray* execution_plan;
  // TF_LITE_ENSURE_STATUS(context->GetExecutionPlan(context, &execution_plan));
  // for (int exec_index = 0; exec_index < execution_plan->size; exec_index++) {
  //    int node_index = execution_plan->data[exec_index];
  //    TfLiteNode* node;
  //    TfLiteRegistration* reg;
  //    context->GetNodeAndRegistration(context, node_index, &node, &reg);
  // }
  // WARNING: This is an experimental interface that is subject to change.
  TfLiteStatus (*GetExecutionPlan)(struct TfLiteContext* context,
                                   TfLiteIntArray** execution_plan);

  // An array of tensors in the interpreter context (of length `tensors_size`)
  TfLiteTensor* tensors;

  // opaque full context ptr (an opaque c++ data structure)
  void* impl_;

  // Request memory pointer be resized. Updates dimensions on the tensor.
  // NOTE: ResizeTensor takes ownership of newSize.
  TfLiteStatus (*ResizeTensor)(struct TfLiteContext*, TfLiteTensor* tensor,
                               TfLiteIntArray* new_size);
  // Request that a error be reported with format string msg.
  void (*ReportError)(struct TfLiteContext*, const char* msg, ...);

  // Add `tensors_to_add` tensors, preserving pre-existing Tensor entries.  If
  // non-null, the value pointed to by `first_new_tensor_index` will be set to
  // the index of the first new tensor.
  TfLiteStatus (*AddTensors)(struct TfLiteContext*, int tensors_to_add,
                             int* first_new_tensor_index);

  // Get a Tensor node by node_index.
  // WARNING: This is an experimental interface that is subject to change.
  TfLiteStatus (*GetNodeAndRegistration)(struct TfLiteContext*, int node_index,
                                         TfLiteNode** node,
                                         TfLiteRegistration** registration);

  // Replace ops with one or more stub delegate operations. This function
  // does not take ownership of `nodes_to_replace`.
  TfLiteStatus (*ReplaceSubgraphsWithDelegateKernels)(
      struct TfLiteContext*, TfLiteRegistration registration,
      const TfLiteIntArray* nodes_to_replace, TfLiteDelegate* delegate);

  // Number of threads that are recommended to subsystems like gemmlowp and
  // eigen.
  int recommended_num_threads;

  // TODO(ahentz): we should create a more general mechanism for this sort of
  // library-global objects.
  void* gemm_context;
  void* eigen_context;
} TfLiteContext;

typedef struct _TfLiteRegistration {
  // Initializes the op from serialized data.
  // If a built-in op:
  //   `buffer` is the op's params data (TfLiteLSTMParams*).
  //   `length` is zero.
  // If custom op:
  //   `buffer` is the op's `custom_options`.
  //   `length` is the size of the buffer.
  //
  // Returns a type-punned (i.e. void*) opaque data (e.g. a primitive pointer
  // or an instance of a struct).
  //
  // The returned pointer will be stored with the node in the `user_data` field,
  // accessible within prepare and invoke functions below.
  // NOTE: if the data is already in the desired format, simply implement this
  // function to return `nullptr` and implement the free function to be a no-op.
  void* (*init)(TfLiteContext* context, const char* buffer, size_t length);

  // The pointer `buffer` is the data previously returned by an init invocation.
  void (*free)(TfLiteContext* context, void* buffer);

  // prepare is called when the inputs this node depends on have been resized.
  // context->ResizeTensor() can be called to request output tensors to be
  // resized.
  //
  // Returns kTfLiteOk on success.
  TfLiteStatus (*prepare)(TfLiteContext* context, TfLiteNode* node);

  // Execute the node (should read node->inputs and output to node->outputs).
  // Returns kTfLiteOk on success.
  TfLiteStatus (*invoke)(TfLiteContext* context, TfLiteNode* node);

  // Builtin codes. If this kernel refers to a builtin this is the code
  // of the builtin. This is so we can do marshaling to other frameworks like
  // NN API. Note, it is the responsibility of the registration binder to
  // set this properly.
  int32_t builtin_code;

  // Custom op name. If the op is a builtin, this will be null.
  // WARNING: This is an experimental interface that is subject to change.
  const char* custom_name;
} TfLiteRegistration;

// WARNING: This is an experimental interface that is subject to change.
typedef struct _TfLiteDelegate {
  // Data that delegate needs to identify itself. This data is owned by the
  // delegate. The delegate is owned in the user code, so the delegate is
  // responsible for doing this when it is destroyed.
  void* data_;

  // Invoked by ModifyGraphWithDelegate. This prepare is called, giving the
  // delegate a view of the current graph through TfLiteContext*. It typically
  // will look at the nodes and call ReplaceSubgraphsWithDelegateKernels()
  // to ask the TensorFlow lite runtime to create macro-nodes to represent
  // delegated subgraphs of the original graph.
  TfLiteStatus (*Prepare)(TfLiteContext* context, TfLiteDelegate* delegate);

  // Copy the data from delegate buffer handle to raw memory.
  // This can be null if the delegate doesn't use its own buffer.
  TfLiteStatus (*CopyFromBufferHandle)(TfLiteDelegate* delegate,
                                       TfLiteBufferHandle buffer_handle,
                                       void* data, int size);

  // Copy the data from raw memory to delegate buffer handle.
  // This can be null if the delegate doesn't use its own buffer.
  TfLiteStatus (*CopyToBufferHandle)(TfLiteDelegate* delegate,
                                     TfLiteBufferHandle buffer_handle,
                                     void* data, int size);

  // Free the Delegate Buffer Handle. Note: This only frees the handle, but
  // this doesn't release the underlying resource (e.g. textures). The
  // resources are either owned by application layer or the delegate.
  // This can be null if the delegate doesn't use its own buffer.
  void (*FreeBufferHandle)(TfLiteDelegate* delegate,
                           TfLiteBufferHandle* handle);
} TfLiteDelegate;

// WARNING: This is an experimental interface that is subject to change.
typedef struct {
  TfLiteDelegate* delegate;
  TfLiteIntArray* nodes_to_replace;
  TfLiteIntArray* input_tensors;
  TfLiteIntArray* output_tensors;
} TfLiteDelegateParams;

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
#endif  // TENSORFLOW_CONTRIB_LITE_CONTEXT_H_
